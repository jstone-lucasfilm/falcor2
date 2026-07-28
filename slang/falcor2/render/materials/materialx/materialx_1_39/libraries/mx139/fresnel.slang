// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Fresnel policies for Mx139 microfacet BSDFs.

implementing mx139;

namespace mx139 {

public float oriented_dielectric_eta(float eta_value, bool backside)
{
    if (backside)
        eta_value = 1.0 / eta_value;
    return eta_value >= 1.0 ? clamp(eta_value, 1.001, 5.0) : clamp(eta_value, 1.0 / 5.0, 1.0 / 1.001);
}

public float dielectric_base_ior(float eta)
{
    return clamp(eta < 1.0 ? 1.0 / eta : eta, 1.001, 5.0);
}

public float dielectric_tir_cos(float eta)
{
    return eta >= 1.0 ? 0.0 : sqrt(1.0 - eta * eta);
}

// BSDL: bsdf_dielectric_impl.h:21-64 DielectricFresnel
[ForceInline]
public float dielectric_fresnel_eval(float eta, float c)
{
    c = clamp(c, 0.0, 1.0);
    float g = (eta - 1.0) * (eta + 1.0) + c * c;
    if (g > 0.0) {
        g = sqrt(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
        return 0.5 * A * A * (1.0 + B * B);
    }
    return 1.0;
}

public float3 schlick_fresnel(float3 F0, float3 F90, float exponent, float c)
{
    return lerp(F0, F90, pow(1.0 - c, exponent));
}

public float3 schlick_color82_fresnel(float3 F0, float3 F82, float3 F90, float exponent, float c)
{
    const float cos_theta_max = 1.0 / 7.0;
    const float cos_theta_factor = 1.0 / (cos_theta_max * pow(1.0 - cos_theta_max, 6.0));
    float one_minus_max = 1.0 - cos_theta_max;
    float one_minus_c = 1.0 - c;
    float one_minus_c6 = one_minus_c * one_minus_c * one_minus_c * one_minus_c * one_minus_c * one_minus_c;
    float3 a = schlick_fresnel(F0, F90, exponent, cos_theta_max) * (float3(1.0) - F82) * cos_theta_factor;
    return schlick_fresnel(F0, F90, exponent, c) - a * c * one_minus_c6;
}

public interface IMxFresnelPolicy
{
    float3 eval(float cos_theta);
}

public interface IMxReflectiveFresnelPolicy : IMxFresnelPolicy
{
    float3 directional_albedo(float cos_theta, float roughness);

    /// Extinction coefficient k of the conductor's complex IOR, for property
    /// reporting only -- nothing in transport consults it.
    ///
    /// Reported because a normal-incidence reflectance cannot carry a conductor's
    /// edge tint: artistic_ior constructs k so that F0 is identically the authored
    /// reflectivity for *any* edge colour, so F0 alone leaves the Fresnel curve
    /// above it undetermined. k closes that gap exactly rather than approximately
    /// -- given the reported F0, n follows, and with (n, k) the whole curve does.
    /// A second reflectance at a fixed grazing angle would also work but freezes
    /// an angle into the reported API and is roughness-contaminated.
    float3 extinction_coefficient();

    /// Refractive index n of the same complex IOR, the other half of the pair.
    /// Both conformers already hold it; reporting it makes the (F0, k) -> n
    /// recovery a read instead of a two-branched inversion.
    float3 index_of_refraction();
}

public interface IMxRefractiveFresnelPolicy : IMxFresnelPolicy
{
    float refraction_eta();

    /// GGX directional albedo of a lobe with this Fresnel, from the quantities
    /// that define its reflectance. `alpha` is the MaterialX-convention average
    /// alpha (microfacet_average_alpha). The specification's layering invariant
    /// relates a layer top's transmittance to this quantity: what a lobe does
    /// not reflect it must hand down, so a transmittance keyed on anything the
    /// reflectance ignores (or that ignores the reflectance) violates it.
    float3 layer_directional_albedo(float cos_theta, float alpha);

    /// True when the authored interface is index-matched (ior exactly 1): no
    /// optical interface exists, so nothing reflects and everything transmits.
    /// This has to be a flag rather than a reading of `refraction_eta`, because
    /// `oriented_dielectric_eta` clamps eta into [1.001, 5] before anything
    /// downstream sees it -- the clamp protects the refraction half-vector
    /// math, whose denominators degenerate at eta = 1, but it also manufactures
    /// a grazing Fresnel sliver that occludes the base of a `layer` and
    /// reflects where the authored material reflects nothing. Real sources use
    /// ior = 1 as a mix background meaning "nothing here", which is why the
    /// exact case is load-bearing.
    bool index_matched();
}

public struct MxConductorFresnel : IMxReflectiveFresnelPolicy
{
    public float3 IOR;
    public float3 extinction;

    public __init(float3 IOR, float3 extinction)
    {
        this.IOR = IOR;
        this.extinction = extinction;
    }

    public __init()
    {
        IOR = float3(1.0);
        extinction = float3(0.0);
    }

    // BSDL: bsdf_conductor_impl.h:19-42 ConductorFresnel::eval
    [ForceInline]
    public float3 eval(float cos_theta)
    {
        cos_theta = clamp(cos_theta, 0.0, 1.0);
        const float3 one = float3(1.0);
        const float3 cos2_theta = float3(cos_theta * cos_theta);
        const float3 sin2_theta = one - cos2_theta;
        const float3 n = IOR;
        const float3 k = extinction;
        const float3 n2 = n * n;
        const float3 k2 = k * k;
        const float3 t0 = n2 - k2 - sin2_theta;
        const float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
        const float3 t1 = a2plusb2 + cos2_theta;
        const float3 a = sqrt(0.5 * (a2plusb2 + t0));
        const float3 t2 = (2.0 * cos_theta) * a;
        const float3 rs = (t1 - t2) / clamp(t1 + t2, float.MIN, BIG);
        const float3 t3 = cos2_theta * a2plusb2 + sin2_theta * sin2_theta;
        const float3 t4 = t2 * sin2_theta;
        const float3 rp = rs * (t3 - t4) / clamp(t3 + t4, float.MIN, BIG);
        return 0.5 * clamp(rp + rs, 0.0, 2.0);
    }

    // BSDL: bsdf_conductor_impl.h:62-76 ConductorFresnel::avg
    //
    // Reports the conductor's normal-incidence reflectance, computed exactly from
    // the complex IOR.
    //
    // This previously evaluated a fitted rational polynomial in (n, k), which is
    // badly behaved for the complex IORs MaterialX's artistic_ior produces. A
    // white edge colour gives n = (1 - r) / (1 + r), below 1 for every
    // reflectivity r, and the fit was evidently not built for that region.
    // Sweeping an authored reflectivity from 0 to 1 and reading the reported value
    // back gave 0.1897 at r = 0.25, a *peak* of 0.2266 near r = 0.69, then 0.2213
    // at r = 1: four times too dark at the bright end, non-monotone, and with the
    // chromaticity almost flattened out of it. A gold authored as
    // [0.944, 0.776, 0.373] came back [0.623, 0.526, 0.450].
    //
    // The formula below is the exact inverse of artistic_ior, so a metal authored
    // with a given reflectivity now reports that reflectivity back. It is the
    // normal-incidence value rather than a hemispherical average, which
    // under-estimates the true average by roughly (1 - F0) / 21 for a Schlick-like
    // grazing rise -- at most ~5%, against the ~4x error it replaces, and unlike
    // the Kulla-Conty average term it stays exactly 0 for a degenerate
    // n = 1, k = 0 interface.
    //
    // Scope of the change: this reaches eval_albedo, hence ExtraBsdfProperties and
    // the encoder inputs, and also the layerable BSDF's directional_albedo, which
    // bsdf_wrapper.slang uses for layer base_scale and for balance_probability. So
    // layered and mixed conductors shade and sample differently than before. That
    // is the intended correction rather than a side effect: a
    // layer(conductor, base) whose conductor reflects 90% previously passed about
    // 78% of its light through to the base instead of about 10%.
    public float3 avg()
    {
        const float3 one = float3(1.0);
        const float3 k2 = extinction * extinction;
        const float3 numerator = (IOR - one) * (IOR - one) + k2;
        const float3 denominator = (IOR + one) * (IOR + one) + k2;
        return clamp(numerator / max(denominator, float3(float.MIN)), 0.0, 1.0);
    }

    [ForceInline]
    public float3 directional_albedo(float cos_theta, float roughness) { return avg(); }

    [ForceInline]
    public float3 extinction_coefficient() { return extinction; }

    [ForceInline]
    public float3 index_of_refraction() { return IOR; }
};

public struct MxConductorAiryFresnel : IMxReflectiveFresnelPolicy
{
    public float3 IOR;
    public float3 extinction;
    public float thinfilm_thickness;
    public float thinfilm_ior;

    public __init(float3 IOR, float3 extinction, float thinfilm_thickness, float thinfilm_ior)
    {
        this.IOR = IOR;
        this.extinction = extinction;
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init()
    {
        IOR = float3(1.0);
        extinction = float3(0.0);
        thinfilm_thickness = 0.0;
        thinfilm_ior = 1.5;
    }

    [ForceInline]
    public float3 extinction_coefficient() { return extinction; }

    [ForceInline]
    public float3 index_of_refraction() { return IOR; }

    [ForceInline]
    public float3 eval(float cos_theta)
    {
        if (thinfilm_thickness <= 0.0)
            return MxConductorFresnel(IOR, extinction).eval(cos_theta);
        MxThinFilmConductorFresnelData fd
            = MxThinFilmConductorFresnelData(IOR, extinction, thinfilm_thickness, thinfilm_ior);
        return fd.compute_fresnel(clamp(cos_theta, 0.0, 1.0));
    }

    [ForceInline]
    public float3 directional_albedo(float cos_theta, float roughness)
    {
        if (thinfilm_thickness <= 0.0)
            return MxConductorFresnel(IOR, extinction).directional_albedo(cos_theta, roughness);
        MxThinFilmConductorFresnelData fd
            = MxThinFilmConductorFresnelData(IOR, extinction, thinfilm_thickness, thinfilm_ior);
        return fd.directional_albedo(clamp(cos_theta, 0.0, 1.0), roughness);
    }
};

public struct MxDielectricFresnel : IMxRefractiveFresnelPolicy
{
    public float eta;
    public bool matched;

    // Takes the raw authored ior: the BSDL boundary clamp (the 1.001 floor)
    // happens here rather than at the construction sites, so the index-matched
    // test can see the authored value before the clamp erases it. The test is
    // "at or below the floor" rather than "exactly 1" because one construction
    // site cannot be moved inside: the genslangpt codegen template inlines
    // clamp_dielectric_ior into generated source, so a sentinel authored as 1
    // arrives from that path already clamped to exactly the floor. Every ior
    // the clamp floors is treated as matched, which is what the clamp already
    // decided physically: F0 at the floor is 6e-8, so the only thing the floor
    // value ever contributed to a layer was the spurious grazing sliver.
    public __init(float eta_value, bool backside)
    {
        matched = eta_value <= OSLE_DIELECTRIC_IOR_MIN;
        eta = oriented_dielectric_eta(clamp_dielectric_ior(eta_value), backside);
    }

    public __init()
    {
        eta = 1.5;
        matched = false;
    }

    [ForceInline]
    public float eval_scalar(float c) { return matched ? 0.0 : dielectric_fresnel_eval(eta, c); }

    [ForceInline]
    public float3 eval(float c) { return float3(eval_scalar(c)); }

    [ForceInline]
    public float refraction_eta() { return eta; }

    [ForceInline]
    public bool index_matched() { return matched; }

    // The ior-derived F0 with a white F90, the shape genglsl's
    // mx_dielectric_bsdf builds its dirAlbedo from. The dielectric BSDF's own
    // layer_opacity stays on the tabulated energy curves -- eta genuinely is
    // its reflectance parameter -- so this exists for interface completeness.
    [ForceInline]
    public float3 layer_directional_albedo(float cos_theta, float alpha)
    {
        if (matched)
            return float3(0.0);
        const float f = (eta - 1.0) / (eta + 1.0);
        return ggx_directional_albedo_analytic(cos_theta, alpha, float3(f * f), float3(1.0));
    }
};

public struct MxDielectricAiryFresnel : IMxRefractiveFresnelPolicy
{
    public float eta;
    public float base_ior;
    public float tir_cos;
    public float thinfilm_thickness;
    public float thinfilm_ior;
    public bool matched;

    // Takes the raw authored ior; see MxDielectricFresnel for why the clamp
    // lives inside the constructor and why the matched test is a floor test.
    public __init(float eta_value, bool backside, float thinfilm_thickness, float thinfilm_ior)
    {
        matched = eta_value <= OSLE_DIELECTRIC_IOR_MIN;
        eta = oriented_dielectric_eta(clamp_dielectric_ior(eta_value), backside);
        base_ior = dielectric_base_ior(eta);
        tir_cos = dielectric_tir_cos(eta);
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init()
    {
        eta = 1.5;
        base_ior = 1.5;
        tir_cos = 0.0;
        thinfilm_thickness = 0.0;
        thinfilm_ior = 1.5;
        matched = false;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        if (index_matched())
            return float3(0.0);
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        if (thinfilm_thickness <= 0.0)
            return float3(dielectric_fresnel_eval(eta, c));

        MxThinFilmDielectricFresnelData fd
            = MxThinFilmDielectricFresnelData(base_ior, thinfilm_thickness, thinfilm_ior);
        return fd.compute_fresnel(c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }

    // An active thin film interferes and reflects even on an index-matched
    // base, so only the film-free case is optically absent.
    [ForceInline]
    public bool index_matched() { return matched && thinfilm_thickness <= 0.0; }

    [ForceInline]
    public float3 layer_directional_albedo(float cos_theta, float alpha)
    {
        if (index_matched())
            return float3(0.0);
        const float f = (eta - 1.0) / (eta + 1.0);
        return ggx_directional_albedo_analytic(cos_theta, alpha, float3(f * f), float3(1.0));
    }
};

// BSDL: bsdf_schlick_impl.h:16-35 SchlickFresnel (extends DielectricFresnel)
public struct MxSchlickFresnel : IMxRefractiveFresnelPolicy
{
    public float3 F0;
    public float3 F90;
    public float exponent;
    public float eta;
    public float tir_cos;

    public __init(float3 F0, float3 F90, float exponent, float eta_value, bool backfacing)
    {
        eta = oriented_dielectric_eta(eta_value, backfacing);
        this.F0 = clamp(F0, 0.0, 1.0);
        this.F90 = clamp(F90, 0.0, 1.0);
        this.exponent = exponent;
        tir_cos = dielectric_tir_cos(eta);
    }

    public __init()
    {
        F0 = float3(0.04);
        F90 = float3(1.0);
        exponent = 5.0;
        eta = 1.5;
        tir_cos = 0.0;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        return schlick_fresnel(F0, F90, exponent, c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }

    // A Schlick lobe's reflectance is authored through its colours, not an
    // ior, so the index-matched sentinel does not apply; zeroed F0/F90 is the
    // authored way to disable it, and layer_directional_albedo already reads
    // exactly zero there.
    [ForceInline]
    public bool index_matched() { return false; }

    // The colours that define this lobe's reflectance, through the same
    // analytic albedo fit genglsl's mx_generalized_schlick_bsdf uses. The
    // derived eta plays no part: it exists for the refraction path, and a
    // transmittance keyed on it never consults F0/F90 -- a lobe reflecting
    // nothing would still take energy from everything beneath it.
    [ForceInline]
    public float3 layer_directional_albedo(float cos_theta, float alpha)
    {
        return ggx_directional_albedo_analytic(cos_theta, alpha, F0, F90);
    }
};

public struct MxSchlickColor82Fresnel : IMxRefractiveFresnelPolicy
{
    public float3 F0;
    public float3 F82;
    public float3 F90;
    public float exponent;
    public float eta;
    public float tir_cos;

    public __init(float3 F0, float3 F82, float3 F90, float exponent, float eta_value, bool backfacing)
    {
        eta = oriented_dielectric_eta(eta_value, backfacing);
        this.F0 = clamp(F0, 0.0, 1.0);
        this.F82 = max(F82, float3(0.0));
        this.F90 = clamp(F90, 0.0, 1.0);
        this.exponent = exponent;
        tir_cos = dielectric_tir_cos(eta);
    }

    public __init()
    {
        F0 = float3(0.04);
        F82 = float3(1.0);
        F90 = float3(1.0);
        exponent = 5.0;
        eta = 1.5;
        tir_cos = 0.0;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        return schlick_color82_fresnel(F0, F82, F90, exponent, c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }

    [ForceInline]
    public bool index_matched() { return false; }

    // F82 is deliberately absent: mx_ggx_dir_albedo takes only F0 and F90, and
    // MaterialX's hoffman-schlick correction vanishes when both are zero, so
    // zeroed F0/F90 must read as a fully transparent top whatever color82 says.
    [ForceInline]
    public float3 layer_directional_albedo(float cos_theta, float alpha)
    {
        return ggx_directional_albedo_analytic(cos_theta, alpha, F0, F90);
    }
};

public struct MxSchlickAiryFresnel : IMxRefractiveFresnelPolicy
{
    public float3 F0;
    public float3 F82;
    public float3 F90;
    public float exponent;
    public float eta;
    public float tir_cos;
    public float thinfilm_thickness;
    public float thinfilm_ior;

    public __init(
        float3 F0,
        float3 F82,
        float3 F90,
        float exponent,
        float eta_value,
        bool backfacing,
        float thinfilm_thickness,
        float thinfilm_ior
    )
    {
        eta = oriented_dielectric_eta(eta_value, backfacing);
        this.F0 = clamp(F0, 0.0, 1.0);
        this.F82 = max(F82, float3(0.0));
        this.F90 = clamp(F90, 0.0, 1.0);
        this.exponent = exponent;
        tir_cos = dielectric_tir_cos(eta);
        this.thinfilm_thickness = thinfilm_thickness;
        this.thinfilm_ior = thinfilm_ior;
    }

    public __init()
    {
        F0 = float3(0.04);
        F82 = float3(1.0);
        F90 = float3(1.0);
        exponent = 5.0;
        eta = 1.5;
        tir_cos = 0.0;
        thinfilm_thickness = 0.0;
        thinfilm_ior = 1.5;
    }

    [ForceInline]
    public float3 eval(float c)
    {
        c = clamp(c, 0.0, 1.0);
        if (c < tir_cos)
            return float3(1.0);
        MxThinFilmSchlickFresnelData fd
            = MxThinFilmSchlickFresnelData(F0, F82, F90, exponent, thinfilm_thickness, thinfilm_ior);
        return fd.compute_fresnel(c);
    }

    [ForceInline]
    public float refraction_eta() { return eta; }

    [ForceInline]
    public bool index_matched() { return false; }

    // The authored F0/F90, as genglsl uses regardless of thin film. Whether an
    // active film's chromatic Fresnel should instead flow through here is an
    // open spec question (mx_dielectric_bsdf keeps its dirAlbedo chromatic,
    // mx_generalized_schlick_bsdf averages it); until settled, match genglsl.
    [ForceInline]
    public float3 layer_directional_albedo(float cos_theta, float alpha)
    {
        return ggx_directional_albedo_analytic(cos_theta, alpha, F0, F90);
    }
};

} // namespace mx139
