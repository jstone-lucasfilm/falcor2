// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "microfacet_bsdf.h"

#include "../codegen_support/snippet_loader.h"
#include "../layering_analysis.h"
#include "../policy.h"
#include "../codegen_user_data.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Syntax.h>

#include <fmt/format.h>

#include <memory>
#include <optional>
#include <string>

namespace falcor {
namespace mtlx {
namespace genslangpt {

namespace {

using namespace codegen_support;

class GenslangPtMicrofacetBsdfNode : public mx::ShaderNodeImpl {
public:
    explicit GenslangPtMicrofacetBsdfNode(MxMicrofacetKind kind)
        : m_kind(kind)
    {
    }

    void emitFunctionDefinition(const mx::ShaderNode&, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        auto user_data = context.getUserData<CodegenUserData>("mtlx");
        FALCOR_CHECK(user_data, "MTLX microfacet BSDF node missing user data.");
        if (user_data->emit_state.direct_microfacet_helpers_emitted)
            return;
        user_data->emit_state.direct_microfacet_helpers_emitted = true;

        ensure_snippet_emitted(stage, user_data->emit_state, "mx_direct_microfacet_helpers.slang");
    }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        auto user_data = context.getUserData<CodegenUserData>("mtlx");
        FALCOR_CHECK(user_data && user_data->inputs.desc, "MTLX microfacet BSDF node missing user data.");

        emitOutputVariables(node, context, stage);
        const mx::ShaderOutput* output = node.getOutput();
        FALCOR_CHECK(output, "MTLX microfacet BSDF node missing output.");

        const mx::ShaderGenerator& shadergen = context.getShaderGenerator();
        const std::string output_var = output->getVariable();
        std::string weight = input_expr(node, "weight", context);
        const std::string normal = input_expr(node, "normal", context);
        const std::string tangent = input_expr(node, "tangent", context);
        const bool continues = transmission_continues_to_base(*user_data, node);
        const std::string bsdf_expr = bsdf_constructor_expr(node, context, *user_data->inputs.desc, continues);
        if (continues) {
            // A transmission-only leaf over a BSDF base is pass-through and
            // nothing else, so it is emitted at weight 0: its layer opacity is
            // then 0 on both layering paths and the base receives everything,
            // which a leaf with no lobe at all would not give (bsdf_mix would read
            // its zero transmission as full opacity). An RT leaf keeps its weight
            // and loses only its transmission tint (bsdf_constructor_expr).
            const MxMicrofacetSelection selection = microfacet_selection(*user_data, node, *user_data->inputs.desc);
            const std::string scatter_mode = input_expr(node, "scatter_mode", context);
            if (selection.scatter == MxScatterSelection::static_t)
                weight = "0.0";
            else if (selection.scatter == MxScatterSelection::runtime)
                weight = "((" + scatter_mode + ") == 1 ? 0.0 : " + weight + ")";
        }

        shadergen.emitLine(output_var + ".set_bsdf(mx_stack_next_bsdf++)", stage);
        shadergen.emitLine("mx_stack.bsdf_weights[" + output_var + ".get_index()] = float3(" + weight + ")", stage);
        shadergen.emitLine("mx_stack.set_bsdf(" + output_var + ".get_index(), " + bsdf_expr + ")", stage);
        shadergen.emitLine("direct_prepare_bsdf(" + output_var + ", " + normal + ", " + tangent + ")", stage);
    }

private:
    std::string input_expr(const mx::ShaderNode& node, const std::string& input_name, mx::GenContext& context) const
    {
        const mx::ShaderInput* input = node.getInput(input_name);
        FALCOR_CHECK(
            input,
            "MTLX microfacet BSDF node '{}' is missing nodedef input '{}'.",
            node.getName(),
            input_name
        );
        return context.getShaderGenerator().getUpstreamResult(input, context);
    }

    std::string scatter_reflection_tint(
        MxScatterSelection selection,
        const std::string& tint,
        const std::string& scatter_mode
    ) const
    {
        switch (selection) {
        case MxScatterSelection::runtime:
            return "((" + scatter_mode + ") == 1 ? float3(0.0) : " + tint + ")";
        case MxScatterSelection::static_r:
        case MxScatterSelection::static_rt:
            return tint;
        case MxScatterSelection::static_t:
            return "float3(0.0)";
        }
        FALCOR_UNREACHABLE();
    }

    std::string scatter_transmission_tint(
        MxScatterSelection selection,
        const std::string& tint,
        const std::string& scatter_mode
    ) const
    {
        switch (selection) {
        case MxScatterSelection::runtime:
            return "((" + scatter_mode + ") == 0 ? float3(0.0) : " + tint + ")";
        case MxScatterSelection::static_t:
        case MxScatterSelection::static_rt:
            return tint;
        case MxScatterSelection::static_r:
            return "float3(0.0)";
        }
        FALCOR_UNREACHABLE();
    }

    // Whether the layering analysis placed this node's transmission over another
    // BSDF: a transmissive lobe in the top branch of a layer whose base is a BSDF.
    // The specification's layering model passes such a lobe's transmission to the
    // base, which every layer combiner already does through the top's
    // layer_opacity (1 - E_R of its reflection lobes); the lobe itself therefore
    // contributes no BTDF of its own. Emitting one as well transmitted the same
    // light twice: once to the base as pass-through, and once through the
    // surface into the object, which a path tracer then follows through closed
    // geometry. Every LamaDielectric coat is this case.
    bool transmission_continues_to_base(const CodegenUserData& user_data, const mx::ShaderNode& node) const
    {
        for (const LayeringDesc::BsdfDesc& bsdf : user_data.analysis.layering.bsdfs) {
            if (bsdf.shader_node == &node)
                return bsdf.transmissive && !bsdf.through_material_transmissive;
        }
        return false;
    }

    std::optional<MxMicrofacetSelection>
    cached_microfacet_selection(const CodegenUserData& user_data, const mx::ShaderNode& node) const
    {
        for (const LayeringDesc::BsdfDesc& bsdf : user_data.analysis.layering.bsdfs) {
            if (bsdf.shader_node != &node || !bsdf.has_microfacet_selection)
                continue;

            MxMicrofacetSelection selection;
            selection.kind = m_kind;
            selection.fresnel = static_cast<MxFresnelSelection>(bsdf.fresnel_selection);
            selection.scatter = static_cast<MxScatterSelection>(bsdf.scatter_selection);
            return selection;
        }
        return {};
    }

    std::string direct_constructor_expr(
        const mx::ShaderNode& node,
        mx::GenContext& context,
        const CodeGenDesc& desc,
        MxMicrofacetSelection selection,
        bool transmission_continues
    ) const
    {
        const std::string type = microfacet_bsdf_type(desc, selection);
        const std::string rough = input_expr(node, "roughness", context);
        const std::string retroreflective = input_expr(node, "retroreflective", context);
        const std::string tf_t = input_expr(node, "thinfilm_thickness", context);
        const std::string tf_ior = input_expr(node, "thinfilm_ior", context);

        switch (selection.kind) {
        case MxMicrofacetKind::dielectric: {
            const std::string tint = input_expr(node, "tint", context);
            const std::string scatter_mode = input_expr(node, "scatter_mode", context);
            const std::string ior = input_expr(node, "ior", context);
            const std::string r_tint = scatter_reflection_tint(selection.scatter, tint, scatter_mode);
            const std::string t_tint = transmission_continues
                ? std::string("float3(0.0)")
                : scatter_transmission_tint(selection.scatter, tint, scatter_mode);
            // The raw authored ior goes through: the Fresnel constructors apply
            // the BSDL boundary clamp internally, and they need the pre-clamp
            // value to recognize an index-matched (ior = 1) interface. Their
            // matched test also admits the clamp floor itself, so generated
            // code from a build predating this change behaves identically.
            if (selection.fresnel == MxFresnelSelection::airy)
                return fmt::format(
                    "{type}({rough}, MxDielectricAiryFresnel({ior}, !si.front_facing,"
                    " {tf_t}, {tf_ior}), {r_tint}, {t_tint}, float3(0.0), {retroreflective}, !si.front_facing)",
                    fmt::arg("type", type),
                    fmt::arg("rough", rough),
                    fmt::arg("ior", ior),
                    fmt::arg("tf_t", tf_t),
                    fmt::arg("tf_ior", tf_ior),
                    fmt::arg("r_tint", r_tint),
                    fmt::arg("t_tint", t_tint),
                    fmt::arg("retroreflective", retroreflective)
                );
            return fmt::format(
                "{type}({rough}, MxDielectricFresnel({ior}, !si.front_facing),"
                " {r_tint}, {t_tint}, float3(0.0), {retroreflective}, !si.front_facing)",
                fmt::arg("type", type),
                fmt::arg("rough", rough),
                fmt::arg("ior", ior),
                fmt::arg("r_tint", r_tint),
                fmt::arg("t_tint", t_tint),
                fmt::arg("retroreflective", retroreflective)
            );
        }
        case MxMicrofacetKind::conductor: {
            const std::string ior = input_expr(node, "ior", context);
            const std::string extinction = input_expr(node, "extinction", context);
            if (selection.fresnel == MxFresnelSelection::airy)
                return fmt::format(
                    "{type}({rough}, MxConductorAiryFresnel({ior}, {extinction}, {tf_t}, {tf_ior}), {retroreflective})",
                    fmt::arg("type", type),
                    fmt::arg("rough", rough),
                    fmt::arg("ior", ior),
                    fmt::arg("extinction", extinction),
                    fmt::arg("tf_t", tf_t),
                    fmt::arg("tf_ior", tf_ior),
                    fmt::arg("retroreflective", retroreflective)
                );
            return fmt::format(
                "{type}({rough}, MxConductorFresnel({ior}, {extinction}), {retroreflective})",
                fmt::arg("type", type),
                fmt::arg("rough", rough),
                fmt::arg("ior", ior),
                fmt::arg("extinction", extinction),
                fmt::arg("retroreflective", retroreflective)
            );
        }
        case MxMicrofacetKind::generalized_schlick: {
            const std::string scatter_mode = input_expr(node, "scatter_mode", context);
            const std::string color0 = input_expr(node, "color0", context);
            const std::string color82 = input_expr(node, "color82", context);
            const std::string color90 = input_expr(node, "color90", context);
            const std::string exponent = input_expr(node, "exponent", context);
            const std::string r_tint = scatter_reflection_tint(selection.scatter, "float3(1.0)", scatter_mode);
            const std::string t_tint = transmission_continues
                ? std::string("float3(0.0)")
                : scatter_transmission_tint(selection.scatter, "float3(1.0)", scatter_mode);
            if (selection.fresnel == MxFresnelSelection::airy)
                return fmt::format(
                    "{type}({rough}, make_schlick_airy_fresnel({color0}, {color82}, {color90}, {exponent},"
                    " !si.front_facing, {tf_t}, {tf_ior}), {r_tint}, {t_tint}, {retroreflective},"
                    " !si.front_facing)",
                    fmt::arg("type", type),
                    fmt::arg("rough", rough),
                    fmt::arg("color0", color0),
                    fmt::arg("color82", color82),
                    fmt::arg("color90", color90),
                    fmt::arg("exponent", exponent),
                    fmt::arg("tf_t", tf_t),
                    fmt::arg("tf_ior", tf_ior),
                    fmt::arg("r_tint", r_tint),
                    fmt::arg("t_tint", t_tint),
                    fmt::arg("retroreflective", retroreflective)
                );
            if (selection.fresnel == MxFresnelSelection::color82)
                return fmt::format(
                    "{type}({rough}, make_schlick_color82_fresnel({color0}, {color82}, {color90}, {exponent},"
                    " !si.front_facing), {r_tint}, {t_tint}, {retroreflective}, !si.front_facing)",
                    fmt::arg("type", type),
                    fmt::arg("rough", rough),
                    fmt::arg("color0", color0),
                    fmt::arg("color82", color82),
                    fmt::arg("color90", color90),
                    fmt::arg("exponent", exponent),
                    fmt::arg("r_tint", r_tint),
                    fmt::arg("t_tint", t_tint),
                    fmt::arg("retroreflective", retroreflective)
                );
            return fmt::format(
                "{type}({rough}, make_schlick_fresnel({color0}, {color90}, {exponent}, !si.front_facing),"
                " {r_tint}, {t_tint}, {retroreflective}, !si.front_facing)",
                fmt::arg("type", type),
                fmt::arg("rough", rough),
                fmt::arg("color0", color0),
                fmt::arg("color90", color90),
                fmt::arg("exponent", exponent),
                fmt::arg("r_tint", r_tint),
                fmt::arg("t_tint", t_tint),
                fmt::arg("retroreflective", retroreflective)
            );
        }
        case MxMicrofacetKind::none:
            break;
        }
        FALCOR_UNREACHABLE();
    }

    MxMicrofacetSelection
    microfacet_selection(const CodegenUserData& user_data, const mx::ShaderNode& node, const CodeGenDesc& desc) const
    {
        std::optional<MxMicrofacetSelection> cached = cached_microfacet_selection(user_data, node);
        return cached ? *cached : select_microfacet(desc, node.getImplementation().getName(), &node);
    }

    std::string bsdf_constructor_expr(
        const mx::ShaderNode& node,
        mx::GenContext& context,
        const CodeGenDesc& desc,
        bool transmission_continues
    ) const
    {
        auto user_data = context.getUserData<CodegenUserData>("mtlx");
        std::optional<MxMicrofacetSelection> cached
            = user_data ? cached_microfacet_selection(*user_data, node) : std::optional<MxMicrofacetSelection>{};
        MxMicrofacetSelection selection
            = cached ? *cached : select_microfacet(desc, node.getImplementation().getName(), &node);
        return direct_constructor_expr(node, context, desc, selection, transmission_continues);
    }

    MxMicrofacetKind m_kind = MxMicrofacetKind::none;
};

} // namespace

mx::ShaderNodeImplPtr create_microfacet_dielectric_node()
{
    return std::make_shared<GenslangPtMicrofacetBsdfNode>(MxMicrofacetKind::dielectric);
}

mx::ShaderNodeImplPtr create_microfacet_conductor_node()
{
    return std::make_shared<GenslangPtMicrofacetBsdfNode>(MxMicrofacetKind::conductor);
}

mx::ShaderNodeImplPtr create_microfacet_generalized_schlick_node()
{
    return std::make_shared<GenslangPtMicrofacetBsdfNode>(MxMicrofacetKind::generalized_schlick);
}

} // namespace genslangpt
} // namespace mtlx
} // namespace falcor
