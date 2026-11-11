// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
import falcor2.render;
import falcor2.utils;

static const int TEST_LEAF_COUNT = 4;

public struct SharedBsdfInputProbeResult
{
    uint bsdf_count = {};
    float3 bsdf_weight[TEST_LEAF_COUNT] = {};
    float bsdf_kind[TEST_LEAF_COUNT] = {};
    float3 eval = {};
};

SurfaceInteraction make_surface_interaction(float2 uv, float3 wi)
{
    SurfaceInteraction si = {};
    si.position_ws = float3(uv, 0.0);
    si.wi_ws = wi;
    si.uv = uv;
    si.shading_frame_ws = Frame(float3(0.0, 1.0, 0.0));
    si.normal_ws = si.shading_frame_ws.normal;
    si.front_facing = dot(si.wi_ws, si.normal_ws) >= 0.0;
    si.exterior_ior = 1.0;
    return si;
}

public SharedBsdfInputProbeResult
shared_bsdf_input_probe(uint3 seeds, uint32_t material_id, float2 uv, float3 wi, float3 wo)
{
    SharedBsdfInputProbeResult result = {};
    const SurfaceInteraction si = make_surface_interaction(uv, wi);
    const ExplicitLodSampler lod_sampler = ExplicitLodSampler(0.0);
    let material = g_scene.get_material(MaterialID(material_id));
    let instance = material.setup_material_instance(si, lod_sampler, MaterialInstanceHints::none);

    if (let properties = instance.collect_extra_bsdf_properties(si, wo)) {
        result.bsdf_count = properties.bsdf_count;
        [ForceUnroll]
        for (int i = 0; i < TEST_LEAF_COUNT; ++i) {
            result.bsdf_weight[i] = properties.bsdf_weight[i];
            result.bsdf_kind[i] = properties.bsdf_kind[i];
        }
    }

    // Evaluation is deterministic; ``seeds`` only keeps the probe's batch shape.
    result.eval = instance.eval(si, wo);
    return result;
}
