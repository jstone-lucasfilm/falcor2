// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "microfacet_bsdf.h"

#include "../user_data.h"
#include "bsdf_emission.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <fmt/format.h>

#include <memory>
#include <string>
#include <vector>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

namespace {

class GenslangPtMicrofacetBsdfNode final : public mx::ShaderNodeImpl {
public:
    explicit GenslangPtMicrofacetBsdfNode(MxMicrofacetKind kind)
        : m_kind(kind)
    {
    }

    void emitFunctionDefinition(const mx::ShaderNode&, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        emit_bsdf_preparation_helper(context, stage);
    }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
        FALCOR_CHECK(user_data, "Structured MaterialX microfacet node has no request.");
        emitOutputVariables(node, context, stage);
        const mx::ShaderOutput* output = node.getOutput();
        FALCOR_CHECK(output, "Structured MaterialX microfacet node has no output.");
        const mx::ShaderGenerator& target = context.getShaderGenerator();
        const std::string output_name = output->getVariable();
        const std::string normal = input(node, "normal", context);
        const std::string tangent = input(node, "tangent", context);
        const auto found = user_data->node_emission_facts().microfacet_selections.find(&node);
        FALCOR_CHECK(
            found != user_data->node_emission_facts().microfacet_selections.end(),
            "Structured MaterialX microfacet node '{}' has no resolved emission policy.",
            node.getName()
        );
        // emit_bsdf_initialization assigns the BSDF index before it evaluates
        // the weight and the constructor, so both may test it.
        const TransmissionContinues continues
            = transmission_continues_to_base(user_data->node_emission_facts(), node, output_name + ".get_index()");
        std::string weight = input(node, "weight", context);
        if (continues.any && m_kind != MxMicrofacetKind::conductor) {
            // A transmission-only leaf over a BSDF base is pass-through and
            // nothing else, so it is emitted at weight 0: its layer opacity is
            // then 0 on both layering paths and the base receives everything,
            // which a leaf with no lobe at all would not give (bsdf_mix would read
            // its zero transmission as full opacity). An RT leaf keeps its weight
            // and loses only its transmission tint (constructor).
            const std::string scatter = input(node, "scatter_mode", context);
            std::string coat_weight = weight;
            if (found->second.scatter == MxScatterSelection::static_t)
                coat_weight = "0.0";
            else if (found->second.scatter == MxScatterSelection::runtime)
                coat_weight = "((" + scatter + ") == 1 ? 0.0 : " + weight + ")";
            weight
                = continues.all ? coat_weight : "(" + continues.condition + " ? " + coat_weight + " : " + weight + ")";
        }
        emit_bsdf_initialization(
            target,
            stage,
            output_name,
            weight,
            constructor(node, context, found->second, continues),
            normal,
            tangent
        );
    }

private:
    static std::string input(const mx::ShaderNode& node, const std::string& name, mx::GenContext& context)
    {
        const mx::ShaderInput* value = node.getInput(name);
        FALCOR_CHECK(value, "Structured MaterialX microfacet node '{}' lacks input '{}'.", node.getName(), name);
        return context.getShaderGenerator().getUpstreamResult(value, context);
    }

    static std::string
    reflection_tint(MxScatterSelection selection, const std::string& tint, const std::string& scatter)
    {
        switch (selection) {
        case MxScatterSelection::runtime:
            return "((" + scatter + ") == 1 ? float3(0.0) : " + tint + ")";
        case MxScatterSelection::static_r:
        case MxScatterSelection::static_rt:
            return tint;
        case MxScatterSelection::static_t:
            return "float3(0.0)";
        }
        FALCOR_UNREACHABLE();
    }

    static std::string
    transmission_tint(MxScatterSelection selection, const std::string& tint, const std::string& scatter)
    {
        switch (selection) {
        case MxScatterSelection::runtime:
            return "((" + scatter + ") == 0 ? float3(0.0) : " + tint + ")";
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
    //
    // The verdict belongs to each *leaf* that names the node, not to the node: a
    // nodegraph used twice (two LamaDielectrics, say) is emitted as one shared
    // function whose ShaderNode both leaves report, and one can be a coat while
    // the other is the glass beneath it. Leaves are told apart at runtime by
    // their BSDF index, which follows the layering order.
    struct TransmissionContinues {
        // Expression true where this leaf's transmission continues to a base:
        // a literal when every leaf agrees, else a test on the BSDF index.
        std::string condition = "false";
        bool any = false;
        bool all = false;
    };

    static TransmissionContinues transmission_continues_to_base(
        const NodeEmissionFacts& facts,
        const mx::ShaderNode& node,
        const std::string& index
    )
    {
        std::vector<std::size_t> continuing;
        std::size_t leaves = 0;
        for (std::size_t i = 0; i < facts.bsdf_leaves.size(); ++i) {
            if (facts.bsdf_leaves[i].shader_node != &node)
                continue;
            ++leaves;
            if (facts.bsdf_leaves[i].transmission_continues_to_base)
                continuing.push_back(i);
        }
        TransmissionContinues result;
        result.any = !continuing.empty();
        result.all = result.any && continuing.size() == leaves;
        if (!result.any || result.all) {
            result.condition = result.all ? "true" : "false";
            return result;
        }
        result.condition.clear();
        for (std::size_t i = 0; i < continuing.size(); ++i)
            result.condition += fmt::format("{}({} == {})", i == 0 ? "(" : " || ", index, continuing[i]);
        result.condition += ")";
        return result;
    }

    // The transmission tint a leaf is constructed with: zero where its
    // transmission continues to a base, the authored tint where it leaves the
    // material, and a runtime choice where the shared function serves both.
    static std::string continuing_transmission_tint(const TransmissionContinues& continues, const std::string& tint)
    {
        if (!continues.any)
            return tint;
        if (continues.all)
            return "float3(0.0)";
        return "(" + continues.condition + " ? float3(0.0) : " + tint + ")";
    }

    std::string constructor(
        const mx::ShaderNode& node,
        mx::GenContext& context,
        MxMicrofacetSelection selection,
        const TransmissionContinues& continues
    ) const
    {
        FALCOR_CHECK(selection.kind == m_kind, "Structured MaterialX microfacet emitter received the wrong family.");
        const std::string type = microfacet_bsdf_type(selection);
        const std::string rough = input(node, "roughness", context);
        const std::string retroreflective = input(node, "retroreflective", context);
        const std::string thickness = input(node, "thinfilm_thickness", context);
        const std::string thinfilm_ior = input(node, "thinfilm_ior", context);
        if (m_kind == MxMicrofacetKind::dielectric) {
            const std::string tint = input(node, "tint", context);
            const std::string scatter = input(node, "scatter_mode", context);
            const std::string ior = input(node, "ior", context);
            const std::string reflection = reflection_tint(selection.scatter, tint, scatter);
            const std::string transmission
                = continuing_transmission_tint(continues, transmission_tint(selection.scatter, tint, scatter));
            // The raw authored ior goes through: the Fresnel constructors apply
            // the BSDL boundary clamp internally, and they need the pre-clamp
            // value to recognize an index-matched (ior = 1) interface.
            if (selection.fresnel == MxFresnelSelection::airy)
                return fmt::format(
                    "{}({}, MxDielectricAiryFresnel({}, !si.front_facing, {}, {}), {}, {}, "
                    "float3(0.0), {}, !si.front_facing)",
                    type,
                    rough,
                    ior,
                    thickness,
                    thinfilm_ior,
                    reflection,
                    transmission,
                    retroreflective
                );
            return fmt::format(
                "{}({}, MxDielectricFresnel({}, !si.front_facing), {}, {}, float3(0.0), {}, "
                "!si.front_facing)",
                type,
                rough,
                ior,
                reflection,
                transmission,
                retroreflective
            );
        }
        if (m_kind == MxMicrofacetKind::conductor) {
            if (selection.fresnel != MxFresnelSelection::airy)
                return fmt::format(
                    "{}({}, MxConductorFresnel({}, {}), {})",
                    type,
                    rough,
                    input(node, "ior", context),
                    input(node, "extinction", context),
                    retroreflective
                );
            return fmt::format(
                "{}({}, MxConductorAiryFresnel({}, {}, {}, {}), {})",
                type,
                rough,
                input(node, "ior", context),
                input(node, "extinction", context),
                thickness,
                thinfilm_ior,
                retroreflective
            );
        }
        const std::string scatter = input(node, "scatter_mode", context);
        const std::string color0 = input(node, "color0", context);
        const std::string color82 = input(node, "color82", context);
        const std::string color90 = input(node, "color90", context);
        const std::string exponent = input(node, "exponent", context);
        const std::string reflection = reflection_tint(selection.scatter, "float3(1.0)", scatter);
        const std::string transmission
            = continuing_transmission_tint(continues, transmission_tint(selection.scatter, "float3(1.0)", scatter));
        if (selection.fresnel == MxFresnelSelection::airy)
            return fmt::format(
                "{}({}, make_schlick_airy_fresnel({}, {}, {}, {}, !si.front_facing, {}, {}), {}, {}, {}, "
                "!si.front_facing)",
                type,
                rough,
                color0,
                color82,
                color90,
                exponent,
                thickness,
                thinfilm_ior,
                reflection,
                transmission,
                retroreflective
            );
        if (selection.fresnel == MxFresnelSelection::color82)
            return fmt::format(
                "{}({}, make_schlick_color82_fresnel({}, {}, {}, {}, !si.front_facing), {}, {}, {}, "
                "!si.front_facing)",
                type,
                rough,
                color0,
                color82,
                color90,
                exponent,
                reflection,
                transmission,
                retroreflective
            );
        return fmt::format(
            "{}({}, make_schlick_fresnel({}, {}, {}, !si.front_facing), {}, {}, {}, !si.front_facing)",
            type,
            rough,
            color0,
            color90,
            exponent,
            reflection,
            transmission,
            retroreflective
        );
    }

    MxMicrofacetKind m_kind;
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

} // namespace falcor::mtlx::structured
