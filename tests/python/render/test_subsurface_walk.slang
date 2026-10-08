// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

/// One subsurface walk per thread, entering a scene's geometry at ``entry_ws``
/// through its outward normal ``normal_ws`` along a cosine-weighted direction,
/// as a subsurface lobe samples it. Writes the walk's weight where it left the
/// geometry through a surface facing ``normal_ws`` (the entry face: a
/// reflection), and zero otherwise.
public float3 subsurface_reflection(
    uint thread_id,
    uint seed,
    float3 entry_ws,
    float3 normal_ws,
    float3 albedo,
    float3 radius,
    float anisotropy,
    float ior
)
{
    UniformSampleGenerator sg = UniformSampleGenerator(seed * 1048576u + thread_id);
    SubsurfaceParameters params = {};
    params.albedo = albedo;
    params.radius = radius;
    params.anisotropy = anisotropy;
    params.ior = ior;

    const Frame frame = Frame(-normal_ws);
    const float3 dir = frame.to_global(square_to_cosine_hemisphere_polar(next_float2(sg)));
    RayQuerySceneIntersector intersector = RayQuerySceneIntersector();
    const SubsurfaceExit exit = trace_subsurface_walk(intersector, entry_ws, normal_ws, dir, params, sg);
    if (!exit.valid)
        return float3(0.0);
    const SurfaceInteraction si = g_scene.make_surface_interaction(exit.hit.unpack(), normal_ws);
    return dot(si.normal_ws, normal_ws) > 0.5 ? exit.weight : float3(0.0);
}

/// Chiang's inversion, for the host side to check against its own transcription.
public float3 single_scatter_albedo(float3 albedo)
{
    return subsurface_single_scatter_albedo(albedo);
}
