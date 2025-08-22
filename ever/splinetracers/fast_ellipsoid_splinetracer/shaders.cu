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

#include <optix.h>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <cstdint>
#include "Forward.h"

extern "C" {
    __constant__ Params params;
}

union Union32 
{
    uint32_t u;
    int32_t i;
    float f;
};

__inline__ __device__ uint32_t F32_asuint(float f) { Union32 u; u.f = f; return u.u; }
__inline__ __device__ int32_t F32_asint(float f) { Union32 u; u.f = f; return u.i; }
__inline__ __device__ float U32_asfloat(uint32_t x) { Union32 u; u.u = x; return u.f; }

struct SplineState {
    float logT;
    glm::vec3 C;
    float t;
    glm::vec4 drgb;
};

struct ControlPoint {
    float t;
    glm::vec4 dirac;
};

struct SplineOutput {
    glm::vec3 C;
    float opacity;
};

#define COMPARE_AND_SWAP(t_hit, hit_info, dist_reg, info_reg) \
    do { \
        float old_dist = U32_asfloat(optixGetPayload_##dist_reg()); \
        if (t_hit < old_dist) { \
            uint old_info = optixGetPayload_##info_reg(); \
            optixSetPayload_##dist_reg(F32_asuint(t_hit)); \
            optixSetPayload_##info_reg(hit_info); \
            t_hit = old_dist; \
            hit_info = old_info; \
        } \
    } while(0)
    
__device__ void report_and_sort_hit(float t_hit, uint hit_info) {
    COMPARE_AND_SWAP(t_hit, hit_info,  0,  1); COMPARE_AND_SWAP(t_hit, hit_info,  2,  3);
    COMPARE_AND_SWAP(t_hit, hit_info,  4,  5); COMPARE_AND_SWAP(t_hit, hit_info,  6,  7);
    COMPARE_AND_SWAP(t_hit, hit_info,  8,  9); COMPARE_AND_SWAP(t_hit, hit_info, 10, 11);
    COMPARE_AND_SWAP(t_hit, hit_info, 12, 13); COMPARE_AND_SWAP(t_hit, hit_info, 14, 15);
    COMPARE_AND_SWAP(t_hit, hit_info, 16, 17); COMPARE_AND_SWAP(t_hit, hit_info, 18, 19);
    COMPARE_AND_SWAP(t_hit, hit_info, 20, 21); COMPARE_AND_SWAP(t_hit, hit_info, 22, 23);
    COMPARE_AND_SWAP(t_hit, hit_info, 24, 25); COMPARE_AND_SWAP(t_hit, hit_info, 26, 27);
    COMPARE_AND_SWAP(t_hit, hit_info, 28, 29); COMPARE_AND_SWAP(t_hit, hit_info, 30, 31);
}

__device__ bool intersect_ray_ellipsoid(
    glm::vec3 ray_origin, glm::vec3 ray_direction,
    glm::vec3 mean, glm::vec3 scale, glm::vec4 quat,
    float& t_entry, float& t_exit) 
{
    // レイを楕円体のローカル空間に変換
    glm::vec3 ro = ray_origin - mean;
    const float r = quat.x, x = quat.y, y = quat.z, z = quat.w;
    const glm::mat3 inv_rot = {
        1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + r * z), 2.0f * (x * z - r * y),
        2.0f * (x * y - r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + r * x),
        2.0f * (x * z + r * y), 2.0f * (y * z - r * x), 1.0f - 2.0f * (x * x + y * y)
    };
    ro = inv_rot * ro;
    glm::vec3 rd = inv_rot * ray_direction;
    ro = ro / scale;
    rd = rd / scale;

    // 単位球との交差判定
    float A = glm::dot(rd, rd);
    float B = 2.0f * glm::dot(ro, rd);
    float C = glm::dot(ro, ro) - 1.0f;
    float delta = B * B - 4.0f * A * C;
    if (delta < 0.0f) return false;

    float sqrt_delta = sqrtf(delta);
    t_entry = (-B - sqrt_delta) / (2.0f * A);
    t_exit  = (-B + sqrt_delta) / (2.0f * A);
    return true;
}

__device__ SplineState update(SplineState state, ControlPoint ctrl_pt) {
    const float dt = max(ctrl_pt.t - state.t, 0.f);

    SplineState new_state;
    new_state.drgb = state.drgb + ctrl_pt.dirac;
    new_state.t = ctrl_pt.t;

    glm::vec4 avg_drgb = state.drgb;
    float area = max(avg_drgb.x * dt, 0.f);

    glm::vec3 rgb_norm = glm::vec3(avg_drgb.y, avg_drgb.z, avg_drgb.w) / avg_drgb.x;

    new_state.logT = area + state.logT;
    const float weight = (1.0f - expf(-area)) * expf(-state.logT);
    new_state.C = state.C + weight * rgb_norm;

    return new_state;
}

// spline-machine.slangのextract_color関数を移植
__device__ SplineOutput extract_color(SplineState state) {
    SplineOutput output;
    output.opacity = 1.0f - expf(-state.logT);
    output.C = state.C;
    return output;
}

__device__ __inline__ uchar4 make_color(const glm::vec3& c)
{
    return make_uchar4(
        static_cast<unsigned char>(glm::clamp(c.x, 0.0f, 1.0f) * 255.99f),
        static_cast<unsigned char>(glm::clamp(c.y, 0.0f, 1.0f) * 255.99f),
        static_cast<unsigned char>(glm::clamp(c.z, 0.0f, 1.0f) * 255.99f),
        255
    );
}



extern "C" __global__ void __anyhit__ah()
{
    const uint triangle_id = optixGetPrimitiveIndex();
    const uint gaussian_idx = triangle_id / 20;

    const glm::vec3 mean = ((glm::vec3*)params.means)[gaussian_idx];
    const glm::vec3 scale = ((glm::vec3*)params.scales)[gaussian_idx];
    const glm::vec4 quat = ((glm::vec4*)params.quats)[gaussian_idx];

    float t_entry, t_exit;
    if (intersect_ray_ellipsoid(
        glm::vec3(optixGetWorldRayOrigin().x, optixGetWorldRayOrigin().y, optixGetWorldRayOrigin().z), 
        glm::vec3(optixGetWorldRayDirection().x, optixGetWorldRayDirection().y, optixGetWorldRayDirection().z),
        mean, scale, quat, t_entry, t_exit)) 
    {
        // ステップ2：過去の交差点を無視
        const float cur_t = optixGetRayTmin();
        if (t_exit < cur_t) {
            optixIgnoreIntersection();
            return;
        }

        // ステップ3：ソーティングネットワークに情報を供給
        const uint entry_info = gaussian_idx * 2;
        const uint exit_info  = gaussian_idx * 2 + 1;
        if (t_entry > cur_t) {
            report_and_sort_hit(t_entry, entry_info);
        }
        if (t_exit > cur_t) {
            report_and_sort_hit(t_exit, exit_info);
        }
    }
    optixIgnoreIntersection();
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    
    // params構造体からレイの始点と方向を取得
    glm::vec3 origin = ((glm::vec3*)params.ray_origins.data)[idx.y * params.width + idx.x];
    glm::vec3 direction = ((glm::vec3*)params.ray_directions.data)[idx.y * params.width + idx.x];
    
    // 状態を初期化
    SplineState state = {0.0f, glm::vec3(0.0f), 0.0f, glm::vec4(0.0f)}; // logT, C, t, drgb

    int iter = 0;
    const int MAX_ITERS = params.max_iters;
    const int BUFFER_SIZE = 16;
    const float LOG_CUTOFF = 5.54f;

    // メインのレイマーチングループ
    while (state.logT < LOG_CUTOFF && iter < MAX_ITERS) {
        uint32_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;
        uint32_t p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31;

        // ペイロードを無限遠で初期化
        p0 = p2 = p4 = p6 = p8 = p10 = p12 = p14 = F32_asuint(1e10f);
        p16 = p18 = p20 = p22 = p24 = p26 = p28 = p30 = F32_asuint(1e10f);

        // レイトレーシングを実行し、__anyhit__ahに交差点を収集させる
        optixTrace(
            params.handle,
            make_float3(origin.x, origin.y, origin.z), 
            make_float3(direction.x, direction.y, direction.z),
            state.t, 1e7f, 0.0f, 
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            0, 1, 0,
            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
            p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31
        );

        bool end = false;
        // 収集・ソートされた交差点リストを処理
        for (int i = 0; i < BUFFER_SIZE; i++) {
            ControlPoint ctrl_pt;
            ctrl_pt.t = U32_asfloat(optixGetPayload(i * 2));
            uint hit_info = optixGetPayload(i * 2 + 1);

            if (ctrl_pt.t > 1e9f) {
                end = true;
                break;
            }
            
            // hit_infoからプリミティブIDと入口/出口をデコード
            uint prim_idx = hit_info / 2;
            bool is_entry = (hit_info % 2) != 0;
            
            // ControlPointのdiracを計算
            // (この部分はparamsからdensityやcolorを取得して計算する必要がある)
            ffloat density = ((float*)params.densities.data)[prim_idx];
            glm::vec3 color = ((glm::vec3*)params.colors.data)[prim_idx];
            float dirac_multi = is_entry ? density : -density;
            ctrl_pt.dirac = glm::vec4(dirac_multi, dirac_multi * color.x, dirac_multi * color.y, dirac_multi * color.z);
            // 積分を1ステップ進める
            state = update(state, ctrl_pt);
            iter++;
            if (!(state.logT < LOG_CUTOFF && iter < MAX_ITERS)) break;
        }
        if (end) break;
    }
    
    // 最終的な色を計算して出力
    SplineOutput output = extract_color(state);
    ((uchar4*)params.image_buffer.data)[idx.y * params.width + idx.x] = make_color(output.C);
}

extern "C" __global__ void __miss__ms()
{
    // レイが何にもヒットしなかった場合、背景色（黒）を設定
    optixSetPayload_f(0, 0.0f);
    optixSetPayload_f(1, 0.0f);
    optixSetPayload_f(2, 0.0f);
    optixSetPayload_f(3, 0.0f); 
}

extern "C" __global__ void __closesthit__ch()
{
    // このアーキテクチャでは使用しないため、何もしない
}
