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

#pragma once
#include "glm/glm.hpp"
#include <cuda_fp16.h>
#include <optix.h>
#include <optix_types.h>

// Primitives
const size_t ICOSAHEDRON_VERTICES = 12;
const size_t ICOSAHEDRON_TRIANGLES = 20;

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <typename T> struct StructuredBuffer {
  T *data;
  size_t size;
};


struct HitData {
    float3 scales;
    float3 mean;
    float4 quat;
    float height;
};

struct SplineState//((packed))
{
  float2 distortion_parts;
  float2 cum_sum;
  float3 padding;
  // Spline state
  float t;
  float4 drgb;

  // Volume Rendering State
  float logT;
  float3 C;
};

// Always on GPU
struct Primitives {
  __half *half_attribs;
  float3 *means; 
  float3 *scales; 
  float4 *quats; 
  float *densities; 
  size_t num_prims;
  float *features; 
  size_t feature_size;

  OptixAabb *aabbs;
  size_t prev_alloc_size;
  
  CUdeviceptr d_vertices = 0; // 頂点バッファへのポインタ
  CUdeviceptr d_indices = 0; // インデックス(どの頂点を結んで三角形を作るか)バッファへのポインタ
  size_t num_vertices_per_prim = ICOSAHEDRON_VERTICES; // プリミティブの頂点数(二十面体なら12)
  size_t num_triangles_per_prim = ICOSAHEDRON_TRIANGLES; // プリミティブの三角形数(二十面体なら20)
  size_t d_vertices_size = 0;
  size_t d_indices_size = 0;
};

struct Cam {
    float fx, fy;
    int height;
    int width;
    float3 U, V, W;
    float3 eye;
};
