// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Path-traced subsurface scattering: a random walk inside a closed mesh.
///
/// A subsurface lobe (``BSDFFlags::subsurface``) samples a direction into the
/// surface and hands the path to this walk, which scatters it through a
/// homogeneous medium derived from the lobe's colour and radius until it leaves
/// the mesh, and returns the exit point to continue the path from. The model
/// follows the production path-traced subsurface of [Chiang2016]:
///
/// * The authored colour is the multiple-scattering albedo a semi-infinite slab
///   reflects; it is inverted to the single-scatter albedo of the medium with
///   Chiang's fit, ``alpha = 1 - exp(-5.09406 a + 2.61188 a^2 - 4.31805 a^3)``.
/// * The radius is the mean free path per channel: extinction is its inverse,
///   and scattering is ``alpha * extinction``. The RGB walk samples distances
///   from one channel at a time (a hero channel, picked in proportion to the
///   path weight), and weights by the mixture.
/// * Directions scatter isotropically, or by Henyey-Greenstein for a non-zero
///   anisotropy.
/// * A path leaves the mesh wherever it next meets the surface, through a
///   diffuse exit lobe, except that the entry segment alone may be turned back
///   by total internal reflection at the interface's relative index, as the
///   production integrator applies it (the first segment only).
///
/// [Chiang2016] Chiang, M. J.-Y., Kutz, P., and Burley, B. "Practical and
/// controllable subsurface scattering for production path tracing." ACM
/// SIGGRAPH 2016 Talks.

/// Whether subsurface lobes walk (``MxSubsurfaceBSDF``). A path tracer that handles
/// ``BSDFFlags::subsurface`` samples exports it true; everything else (the
/// reference path tracer, property and BSDF-data kernels) keeps the default and
/// sees the lobe as the diffuse reflection of its colour, as before the walk.
public extern static const bool ENABLE_SUBSURFACE_WALK = false;

/// Parameters of one subsurface lobe, carried from the sampled lobe to the walk.
public struct SubsurfaceParameters
{
    /// Multiple-scattering albedo the medium reflects on a semi-infinite slab.
    public float3 albedo = {};
    /// Mean free path per channel, in scene units.
    public float3 radius = {};
    /// Henyey-Greenstein anisotropy of the phase function.
    public float anisotropy = 0.0;
    /// Relative index of the interface for the entry segment's total internal
    /// reflection test; 1 disables it.
    public float ior = 1.0;
}

/// Chiang's inversion of a multiple-scattering albedo to a single-scatter albedo.
public float3 subsurface_single_scatter_albedo(const float3 albedo)
{
    const float3 a = saturate(albedo);
    const float3 a2 = a * a;
    return 1.0 - exp(-5.09406 * a + 2.61188 * a2 - 4.31805 * a2 * a);
}

/// The homogeneous medium a subsurface lobe's parameters describe.
public HomogeneousMediumProperties subsurface_medium(const SubsurfaceParameters params)
{
    // A zero radius is no scattering at all: an extinction too large to travel.
    const float3 sigma_t = select(params.radius > 0.0, 1.0 / max(params.radius, 1e-30), float3(1e30));
    const float3 sigma_s = subsurface_single_scatter_albedo(params.albedo) * sigma_t;
    HomogeneousMediumProperties medium = {};
    medium.sigma_s = sigma_s;
    medium.sigma_a = sigma_t - sigma_s;
    medium.anisotropy = params.anisotropy;
    return medium;
}

/// Where a walk left the medium, and the weight it carries out.
public struct SubsurfaceExit
{
    /// The surface the walk left through, packed (``PackedHitInfo.unpack()``).
    public PackedHitInfo hit = {};
    /// Throughput multiplier of the walk, colour included.
    public float3 weight = {};
    /// Direction of the walk's last segment, which left the medium through ``hit``.
    public float3 exit_dir_ws = {};
    /// True if the walk reached a surface; false if it was absorbed or terminated.
    public bool valid = false;
}

/// Constants of the walk, as the production integrator sets them.
public static const uint SUBSURFACE_MAX_STEPS = 256;
public static const float SUBSURFACE_ROULETTE_THRESHOLD = 0.2;

/// Run a subsurface walk.
///
/// @param intersector Scene intersector for the walk's segments.
/// @param position_ws Entry point on the surface.
/// @param normal_ws Outward geometric normal at the entry point.
/// @param dir_ws Sampled direction into the surface (``dot(dir_ws, normal_ws) < 0``).
/// @param params The lobe's parameters.
/// @param sg Sample generator.
public SubsurfaceExit trace_subsurface_walk<I : ISceneIntersector, S : ISampleGenerator>(
    I intersector,
    const float3 position_ws,
    const float3 normal_ws,
    float3 dir_ws,
    const SubsurfaceParameters params,
    inout S sg
)
{
    SubsurfaceExit result = {};
    const HomogeneousMediumProperties medium = subsurface_medium(params);
    const HomogeneousVolumeSampler sampler = HomogeneousVolumeSampler(medium);
    const HenyeyGreensteinPhaseFunction phase = HenyeyGreensteinPhaseFunction(params.anisotropy);

    // The walk's own throughput, separate from the path's: the colour is the
    // walk's to produce, so it starts white.
    float3 throughput = float3(1.0);
    // The walk's position, and the origin its rays are traced from. They differ
    // only while the walk sits on a surface (the entry, or a total internal
    // reflection), where the traced ray is offset off the surface to avoid
    // hitting it again. Distances are measured from the position, so the offset
    // counts as distance already travelled, as the production integrator counts
    // its trace bias: at a radius of a few times the offset, an offset taken as
    // a fresh start would begin the walk measurably deep.
    float3 position = position_ws;
    float3 trace_origin = offset_ray_origin(position_ws, -normal_ws);
    bool first_segment = true;

    for (uint step = 0; step < SUBSURFACE_MAX_STEPS; ++step) {
        float distance;
        if (!sampler.sample_distance(throughput, sg, distance))
            return result;

        const IHitInfo hit = intersector.intersect(Ray(trace_origin, dir_ws, 0.0, float.MAX));
        if (!hit.valid) {
            // The mesh is not closed along this direction: the walk is lost.
            return result;
        }
        const SurfaceInteraction si = g_scene.make_surface_interaction(hit, -dir_ws);
        const float hit_distance = length(si.position_ws - position);

        if (distance >= hit_distance) {
            // The segment reaches the surface before scattering.
            throughput *= sampler.eval_surface_weight(throughput, hit_distance);
            // The entry segment may be reflected back in by total internal
            // reflection (the production integrator tests the first segment only).
            const float cos_exit = abs(dot(dir_ws, si.normal_ws));
            if (first_segment && params.ior > 1.0 && cos_exit * cos_exit < 1.0 - 1.0 / (params.ior * params.ior)) {
                const float3 inward = dot(si.normal_ws, dir_ws) > 0.0 ? -si.normal_ws : si.normal_ws;
                position = si.position_ws;
                trace_origin = offset_ray_origin(si.position_ws, inward);
                dir_ws = cosine_direction(inward, next_float2(sg));
                first_segment = true;
                continue;
            }
            result.hit = hit.pack();
            result.weight = throughput;
            result.exit_dir_ws = dir_ws;
            result.valid = all(isfinite(throughput)) && any(throughput > 0.0);
            return result;
        }

        // Scatter inside the medium.
        throughput *= sampler.eval_event_weight(throughput, distance);
        position += dir_ws * distance;
        trace_origin = position;
        PhaseFunctionSample phase_sample;
        if (!phase.sample(-dir_ws, sg, phase_sample))
            return result;
        dir_ws = phase_sample.wo_ws;
        throughput *= phase_sample.weight;
        first_segment = false;

        // Russian roulette on the walk's weight.
        const float importance = max(throughput.x, max(throughput.y, throughput.z));
        if (importance < SUBSURFACE_ROULETTE_THRESHOLD) {
            const float survival = importance / SUBSURFACE_ROULETTE_THRESHOLD;
            if (next_float(sg) >= survival)
                return result;
            throughput /= survival;
        }
    }
    return result;
}

/// A cosine-weighted direction about ``normal``.
float3 cosine_direction(const float3 normal, const float2 u)
{
    const Frame frame = Frame(normal);
    return frame.to_global(square_to_cosine_hemisphere_polar(u));
}
