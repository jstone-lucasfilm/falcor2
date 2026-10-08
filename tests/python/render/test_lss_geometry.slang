// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"

import falcor2.render;

// ----------------------------------------------------------------------------
// intersect_procedural_lss
// ----------------------------------------------------------------------------

// A tapered segment one unit along x at z = -3, radius 0.1 to 0.15 (the cone the
// intersection solves needs distinct end radii; equal ones are a cylinder).
static const float2x4 SEGMENT = float2x4(float4(-0.5, 0, -3, 0.1), float4(0.5, 0, -3, 0.15));

TEST_CASE(intersect_procedural_lss_unit_direction)
{
    float t;
    bool hit = intersect_procedural_lss(float3(0, 0, 0), float3(0, 0, -1), 0.0, 100.0, SEGMENT, t);
    CHECK(hit);
    // The swept surface at x = 0 sits near radius 0.125 (z = -2.875); a tapered
    // sweep's envelope is a little inside the linear radius.
    CHECK_APPROX(t, 2.875f, 1e-3f);
}

TEST_CASE(intersect_procedural_lss_scaled_direction)
{
    // An instance scaled by 0.01 (a stage in centimetres) hands the intersection an
    // object-space direction of length 100, and the ray parameter stays the world
    // one: the hit is the same point, at a hundredth of the parameter.
    float t;
    bool hit = intersect_procedural_lss(float3(0, 0, 0), float3(0, 0, -100), 0.0, 1.0, SEGMENT, t);
    CHECK(hit);
    CHECK_APPROX(t, 0.02875f, 1e-5f);
}

TEST_CASE(intersect_procedural_lss_scaled_direction_respects_t_max)
{
    float t;
    bool hit = intersect_procedural_lss(float3(0, 0, 0), float3(0, 0, -100), 0.0, 0.02, SEGMENT, t);
    CHECK_FALSE(hit);
}
