// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// MaterialX subsurface_bsdf: a path-traced subsurface lobe (render/subsurface_walk.slang),
// with a diffuse lobe of its colour wherever the walk cannot run.

implementing mtlx;

namespace mtlx {

/// MaterialX ``subsurface_bsdf``.
///
/// With a radius, sampling the lobe enters the surface: it returns a
/// cosine-weighted direction into it, flagged ``BSDFFlags::subsurface``, and a
/// path tracer that handles the flag continues the path through
/// ``trace_subsurface_walk`` from where the walk leaves. The lobe evaluates to
/// zero, as light reaches it only through the walk, so next-event estimation
/// skips it. Its sample weight carries no colour: the walk produces it, and
/// the parameters it needs travel with the sample (``BSDFSample.subsurface_*``).
///
/// A zero radius is no subsurface transport, and the lobe is the Lambertian
/// reflection of its colour, the diffuse limit of a vanishing mean free path.
/// That is also the lobe every kernel sees unless it exports
/// ``ENABLE_SUBSURFACE_WALK`` (render/subsurface_walk.slang), so a renderer that
/// does not handle the flag, and every property or BSDF-data query, reproduces
/// the diffuse fallback this lobe replaced.
///
/// Property reports are unchanged by the walk: the authored colour is the
/// albedo a semi-infinite slab of the medium reflects, which is what a
/// consumer of the albedo row expects, and the radius is reported in the
/// extinction row (``RoughnessInformation.extinction``) whether or not the walk
/// runs.
public struct MxSubsurfaceBSDF : ILayerableBSDF
{
    public float3 albedo_value = {};
    public float3 radius = {};
    public float anisotropy = 0.0;

    public __init() { }

    public __init(float3 albedo_value) { this.albedo_value = albedo_value; }

    public __init(float3 albedo_value, float3 radius, float anisotropy)
    {
        this.albedo_value = albedo_value;
        this.radius = max(radius, float3(0.0));
        this.anisotropy = clamp(anisotropy, -0.99, 0.99);
    }

    /// True if sampling this lobe enters the surface.
    public property bool walks
    {
        get {
            return ENABLE_SUBSURFACE_WALK && any(radius > 0.0);
        }
    }

    // ILayerableBSDF


    // TODO: Remove the eval()/eval_pdf() overrides below when
    // https://github.com/shader-slang/slang/issues/12814 is fixed.
    public override float3 eval(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return eval_with_pdf(wi, wo, bc, component).value;
    }

    public override float eval_pdf(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        return eval_with_pdf(wi, wo, bc, component).pdf;
    }

    public BSDFEval eval_with_pdf(
        const float3 wi,
        const float3 wo,
        const BSDFContext bc,
        const BSDFComponent component = BSDFComponent::full_response
    )
    {
        if (walks)
            return BSDFEval();
        MxBsdfSample r = eval_diffuse_reflection_sample(wi, wo, albedo_value, 1.0);
        return BSDFEval(r.pdf > 0.0 ? r.weight * r.pdf : float3(0.0), r.pdf);
    }

    public bool sample(const float3 wi, const BSDFSampleInput input, const BSDFContext bc, out BSDFSample sample)
    {
        sample = {};
        if (wi.z <= 0.0)
            return false;
        const float3 wo = square_to_cosine_hemisphere_concentric(input.u_dir);
        if (walks) {
            // Into the surface; the weight is the lobe's own, before any colour.
            sample.wo = float3(wo.x, wo.y, -wo.z);
            sample.pdf = wo.z * float.INV_PI;
            sample.weight = float3(1.0);
            sample.flags = BSDFFlags::subsurface;
            sample.subsurface_albedo = albedo_value;
            sample.subsurface_radius = radius;
            sample.subsurface_anisotropy = anisotropy;
            return sample.pdf > 0.0;
        }
        MxBsdfSample r = eval_diffuse_reflection_sample(wi, wo, albedo_value, 1.0);
        if (r.pdf <= 0.0)
            return false;
        sample.wo = r.wo;
        sample.pdf = r.pdf;
        sample.weight = r.weight;
        sample.flags = BSDFFlags::diffuse_reflection;
        return true;
    }


    public AlbedoContributions eval_albedo(const float3 wi)
    {
        const float3 reflection = saturate(albedo_value);
        return AlbedoContributions(reflection, float3(1.0) - reflection, float3(0.0), float3(0.0));
    }

    public RoughnessInformation eval_roughness(const float3 wi)
    {
        // Subsurface has no anisotropic roughness control; mark it
        // diffuse-equivalent for IBSDF.eval_roughness consumers.
        RoughnessInformation result = RoughnessInformation(float2(1.0, 1.0), float3(0.0));
        // The radius rides the extinction row (``RoughnessInformation.extinction``),
        // which no other quantity uses for a subsurface leaf.
        result.extinction = radius;
        result.kind = k_bsdf_kind_subsurface;
        return result;
    }

    public float3 directional_albedo(const float3 wi) { return albedo_value; }

    public float3 layer_opacity(const float3 wi) { return float3(1.0); }

    public property float3 ior_as_reflectance
    {
        get {
            return float3(0.0);
        }
    }
};

/// The subsurface lobe at a zero radius: the diffuse fallback alone.
public MxSubsurfaceBSDF create_mx_subsurface_diffuse_fallback(float3 color)
{
    return MxSubsurfaceBSDF(color);
}

public MxSubsurfaceBSDF create_mx_subsurface(float3 color, float3 radius, float anisotropy)
{
    return MxSubsurfaceBSDF(color, radius, anisotropy);
}

} // namespace mtlx
