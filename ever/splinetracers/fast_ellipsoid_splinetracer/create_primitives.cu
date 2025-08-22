// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define TRI_PER_G 8
#define PT_PER_G 6
#include "create_primitives.h"
#include "glm/glm.hpp"
#include "structs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA call (" << #call << " ) failed with error: '"                \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      throw Exception(ss.str().c_str());                                       \
    }                                                                          \
  } while (0)

#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    cudaError_t error = cudaGetLastError();                                    \
    if (error != cudaSuccess) {                                                \
      std::stringstream ss;                                                    \
      ss << "CUDA error on synchronize with error '"                           \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__     \
         << ")\n";                                                             \
      throw Exception(ss.str().c_str());                                       \
    }                                                                          \
  } while (0)

// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW(call)                                               \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA call (" << #call << " ) failed with error: '"         \
                << cudaGetErrorString(error) << "' (" __FILE__ << ":"          \
                << __LINE__ << ")\n";                                          \
      std::terminate();                                                        \
    }                                                                          \
  } while (0)

class Exception : public std::runtime_error {
public:
  Exception(const char *msg) : std::runtime_error(msg) {}
};

__device__ inline float kernelScale(float density, float modulatedMinResponse, uint32_t opts, float kernelDegree) {
  // const float responseModulation = opts & MOGRenderAdaptiveKernelClamping ? density : 1.0f;
  // const float minResponse        = fminf(modulatedMinResponse / responseModulation, 0.97f);
  const float minResponse = fminf(modulatedMinResponse, 0.97f);

  // bump kernel
  if (kernelDegree < 0) {
      const float k  = fabsf(kernelDegree);
      const float s  = 1.0 / powf(3.0, k);
      const float ks = powf((1.f / (logf(minResponse) - 1.f) + 1.f) / s, 1.f / k);
      return ks;
  }

  // linear kernel
  if (kernelDegree == 0) {
      return ((1.0f - minResponse) / 3.0f) / -0.329630334487f;
  }

  /// generalized gaussian of degree b : scaling a = -4.5/3^b
  /// e^{a*|x|^b}
  const float b = kernelDegree;
  const float a = -4.5f / powf(3.0f, static_cast<float>(b));
  /// find distance r (>0) st e^{a*r^b} = minResponse
  /// TODO : reshuffle the math to call powf only once
  return powf(logf(minResponse) / a, 1.0f / b);
}

// constexpr int ICOSAHEDRON_VERTICES = 12;
// constexpr int ICOSAHEDRON_TRIANGLES = 20;
constexpr float goldenRatio = 1.61803398875f;
constexpr float icosaEdge = 1.323169076499215f;
constexpr float icosaVrtScale = 0.5f * icosaEdge;

__constant__ const glm::vec3 ICO_VERTICES[ICOSAHEDRON_VERTICES] = {
    glm::vec3(-1.0f,  goldenRatio, 0.0f), glm::vec3( 1.0f,  goldenRatio, 0.0f), glm::vec3(0.0f, 1.0f, -goldenRatio),
    glm::vec3(-goldenRatio, 0.0f, -1.0f), glm::vec3(-goldenRatio, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, goldenRatio),
    glm::vec3(goldenRatio, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, goldenRatio), glm::vec3(-1.0f, -goldenRatio, 0.0f),
    glm::vec3(0.0f, -1.0f, -goldenRatio), glm::vec3(goldenRatio, 0.0f, -1.0f), glm::vec3(1.0f, -goldenRatio, 0.0f)
};

__constant__ const glm::ivec3 ICO_INDICES[ICOSAHEDRON_TRIANGLES] = {
    glm::ivec3(0, 1, 5), glm::ivec3(0, 5, 4), glm::ivec3(0, 4, 3), glm::ivec3(0, 3, 2), glm::ivec3(0, 2, 1),
    glm::ivec3(11, 10, 6), glm::ivec3(11, 6, 7), glm::ivec3(11, 7, 8), glm::ivec3(11, 8, 9), glm::ivec3(11, 9, 10),
    glm::ivec3(1, 2, 6), glm::ivec3(2, 3, 9), glm::ivec3(3, 4, 8), glm::ivec3(4, 5, 7), glm::ivec3(5, 1, 6),
    glm::ivec3(10, 9, 2), glm::ivec3(9, 8, 3), glm::ivec3(8, 7, 4), glm::ivec3(7, 6, 5), glm::ivec3(6, 2, 10)
};

__global__ void
kern_create_icosahedrons(const glm::vec3 *means, const glm::vec3 *scales,
                           const glm::vec4 *quats, const float *opacities,
                           const size_t num_prims, 
                           const int vertices_per_prim,   
                           const int triangles_per_prim,  
                           glm::vec3 *out_vertices,
                           glm::ivec3 *out_indices) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_prims) return;

    // パラメータαから密度σを計算
    const float alpha = opacities[i];
    const glm::vec3 scl = scales[i];
    const float min_scale = fminf(fminf(scl.x, scl.y), scl.z);
    const float inv_alpha = -logf(fmaxf(1e-10f, 1.0f - alpha));
    const float min_integration_length = fmaxf(1e-10f, min_scale * 2.0f);
    const float sigma = inv_alpha / min_integration_length;
    
    // 3DGRTのメッシュ生成ロジック
    const glm::vec3 center = means[i];
    const glm::vec4 quat = glm::normalize(quats[i]);
    
    // ... (回転行列 rot の計算) ...
    const float r = quat.x;
    const float x = quat.y;
    const float y = quat.z;
    const float z = quat.w;
  
    const glm::mat3 Rt = {
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z),
        2.0 * (x * z + r * y),
  
        2.0 * (x * y + r * z),       1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - r * x),
  
        2.0 * (x * z - r * y),       2.0 * (y * z + r * x),
        1.0 - 2.0 * (x * x + y * y)};
    const glm::mat3 rot = glm::transpose(Rt);
    
    
    const float kernel_s = kernelScale(sigma, 0.01f, 0, 2.0f);
    const glm::vec3 kscl = glm::vec3(kernel_s) * scl * icosaVrtScale;

    // 頂点変形とインデックス生成
    for (int v = 0; v < vertices_per_prim; ++v) {
        glm::vec3 world_v = rot * (ICO_VERTICES[v] * kscl) + center;
        out_vertices[i * vertices_per_prim + v] = world_v;
    }
    for (int t = 0; t < triangles_per_prim; ++t) {
        out_indices[i * triangles_per_prim + t] = ICO_INDICES[t] + (i * vertices_per_prim);
    }
}

void create_icosahedron_primitives(Primitives &prims, const float* opacities_ptr) {
  const size_t block_size = 1024;
  const size_t num_prims = prims.num_prims;

  // プリミティブの情報を設定
  prims.num_vertices_per_prim = ICOSAHEDRON_VERTICES;
  prims.num_triangles_per_prim = ICOSAHEDRON_TRIANGLES;

  // メモリ確保 (変更なし)
  size_t required_vert_size = sizeof(glm::vec3) * prims.num_vertices_per_prim * num_prims;
  if (required_vert_size > prims.d_vertices_size) {
        if (prims.d_vertices_size > 0) {
            CUDA_CHECK(cudaFree((void*)prims.d_vertices));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prims.d_vertices), required_vert_size));
        prims.d_vertices_size = required_vert_size; // 確保したサイズを記録
    }
  size_t required_idx_size = sizeof(glm::ivec3) * prims.num_triangles_per_prim * num_prims;
    if (required_idx_size > prims.d_indices_size) {
        if (prims.d_indices_size > 0) {
            CUDA_CHECK(cudaFree((void*)prims.d_indices));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&prims.d_indices), required_idx_size));
        prims.d_indices_size = required_idx_size; // 確保したサイズを記録
    }

  // CUDAカーネルの起動 (引数を個別に渡す形式に戻す)
  kern_create_icosahedrons<<<(num_prims + block_size - 1) / block_size, block_size>>>(
      (glm::vec3 *)prims.means,
      (glm::vec3 *)prims.scales,
      (glm::vec4 *)prims.quats,
      opacities_ptr,
      prims.num_prims,
      prims.num_vertices_per_prim,
      prims.num_triangles_per_prim,
      (glm::vec3 *)prims.d_vertices,
      (glm::ivec3 *)prims.d_indices
  );
  
  CUDA_SYNC_CHECK();
}
