// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shared_closures.h"

#include <MaterialXCore/Definition.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>

#include <map>
#include <set>
#include <string>
#include <utility>

namespace falcor::mtlx::structured::document_passes {

namespace mx = MaterialX;

namespace {

// Nested implementation graphs are shallow in practice; the bound only keeps a
// malformed recursive definition from looping.
constexpr int MAX_IMPLEMENTATION_DEPTH = 32;

const std::string BSDF_TYPE = "BSDF";

bool is_closure_path_type(const std::string& type)
{
    return type == BSDF_TYPE || type == mx::SURFACE_SHADER_TYPE_STRING || type == mx::MATERIAL_TYPE_STRING;
}

/// The node instance whose implementation graph is being read, with the
/// instance enclosing it; null at the graph that holds the root.
struct Scope {
    mx::NodePtr instance;
    const Scope* parent = nullptr;
};

/// Whether an input carries a connection, following interface names out to the
/// instance whose implementation holds it.
bool is_connected(const mx::InputPtr& input, const Scope* scope)
{
    if (!input)
        return false;
    if (input->hasInterfaceName()) {
        if (!scope)
            return false;
        return is_connected(scope->instance->getInput(input->getInterfaceName()), scope->parent);
    }
    return input->hasNodeName() || input->hasNodeGraphString() || input->hasOutputString();
}

bool flows_into_layer_top(
    const mx::GraphElementPtr& graph,
    const mx::NodePtr& node,
    const std::string& input_name,
    const Scope* scope,
    int depth
);

/// Whether a BSDF entering `node` through `input_name` ends up, within the
/// node's own result, in the top of a layer whose base is a connected BSDF.
bool enters_layer_top_over_base(const mx::NodePtr& node, const std::string& input_name, const Scope* scope, int depth)
{
    if (depth > MAX_IMPLEMENTATION_DEPTH)
        return false;
    if (node->getCategory() == "layer")
        return input_name == "top" && is_connected(node->getInput("base"), scope);
    const mx::NodeDefPtr node_def = node->getNodeDef();
    const mx::InterfaceElementPtr implementation = node_def ? node_def->getImplementation() : nullptr;
    const mx::NodeGraphPtr graph = implementation ? implementation->asA<mx::NodeGraph>() : nullptr;
    if (!graph)
        return false;
    const Scope inner{node, scope};
    for (const mx::NodePtr& child : graph->getNodes()) {
        for (const mx::InputPtr& input : child->getInputs()) {
            if (input->getType() == BSDF_TYPE && input->getInterfaceName() == input_name
                && flows_into_layer_top(graph, child, input->getName(), &inner, depth + 1))
                return true;
        }
    }
    return false;
}

/// Whether a BSDF entering `node` through `input_name` lies in the top of a
/// layer over a base at that node or at any consumer downstream of it in `graph`.
bool flows_into_layer_top(
    const mx::GraphElementPtr& graph,
    const mx::NodePtr& node,
    const std::string& input_name,
    const Scope* scope,
    int depth
)
{
    if (depth > MAX_IMPLEMENTATION_DEPTH)
        return false;
    if (enters_layer_top_over_base(node, input_name, scope, depth))
        return true;
    for (const mx::NodePtr& consumer : graph->getNodes()) {
        for (const mx::InputPtr& input : consumer->getInputs()) {
            if (input->getType() == BSDF_TYPE && input->getNodeName() == node->getName()
                && flows_into_layer_top(graph, consumer, input->getName(), scope, depth + 1))
                return true;
        }
    }
    return false;
}

/// Whether the closure arriving at `input` of a node in the root's graph sits
/// over a BSDF base, given the context of the node itself.
bool input_context(const mx::NodePtr& node, const mx::InputPtr& input, bool beneath)
{
    return beneath || (input->getType() == BSDF_TYPE && enters_layer_top_over_base(node, input->getName(), nullptr, 0));
}

// The contexts a node is reached in, as bits: 1 on its own, 2 over a base.
constexpr int ALONE = 1;
constexpr int OVER_BASE = 2;

void collect_contexts(
    const mx::NodePtr& node,
    bool beneath,
    std::map<mx::NodePtr, int>& contexts,
    std::set<std::pair<mx::NodePtr, bool>>& visited
)
{
    if (!visited.insert({node, beneath}).second)
        return;
    contexts[node] |= beneath ? OVER_BASE : ALONE;
    for (const mx::InputPtr& input : node->getInputs()) {
        if (!is_closure_path_type(input->getType()))
            continue;
        if (const mx::NodePtr upstream = input->getConnectedNode())
            collect_contexts(upstream, input_context(node, input, beneath), contexts, visited);
    }
}

class Splitter {
public:
    explicit Splitter(const std::map<mx::NodePtr, int>& contexts)
        : m_contexts(contexts)
    {
    }

    /// The node that serves `node` in the given context, copying it where the
    /// node is reached in both contexts and this is the one over a base.
    mx::NodePtr specialize(const mx::NodePtr& node, bool beneath)
    {
        const auto key = std::make_pair(node, beneath);
        if (auto found = m_instances.find(key); found != m_instances.end())
            return found->second;
        mx::NodePtr instance = node;
        const auto context = m_contexts.find(node);
        if (beneath && node->getType() == BSDF_TYPE && context != m_contexts.end()
            && context->second == (ALONE | OVER_BASE)) {
            const mx::GraphElementPtr graph = node->getParent()->asA<mx::GraphElement>();
            instance = graph->addNode(
                node->getCategory(),
                graph->createValidChildName(node->getName() + "_over_base"),
                node->getType()
            );
            instance->copyContentFrom(node);
            ++m_added;
        }
        m_instances[key] = instance;
        for (const mx::InputPtr& input : instance->getInputs()) {
            if (!is_closure_path_type(input->getType()))
                continue;
            const mx::NodePtr upstream = input->getConnectedNode();
            if (!upstream)
                continue;
            const mx::NodePtr specialized = specialize(upstream, input_context(instance, input, beneath));
            if (specialized != upstream)
                input->setNodeName(specialized->getName());
        }
        return instance;
    }

    std::size_t added() const { return m_added; }

private:
    const std::map<mx::NodePtr, int>& m_contexts;
    std::map<std::pair<mx::NodePtr, bool>, mx::NodePtr> m_instances;
    std::size_t m_added = 0;
};

} // namespace

std::size_t split_shared_closures_by_layering_context(mx::Document& document, const mx::TypedElementPtr& root)
{
    mx::NodePtr root_node = root ? root->asA<mx::Node>() : nullptr;
    if (!root_node && root) {
        if (const mx::OutputPtr output = root->asA<mx::Output>())
            root_node = output->getConnectedNode();
    }
    if (!root_node || root_node->getDocument().get() != &document)
        return 0;

    std::map<mx::NodePtr, int> contexts;
    std::set<std::pair<mx::NodePtr, bool>> visited;
    collect_contexts(root_node, false, contexts, visited);
    bool any_conflict = false;
    for (const auto& [node, bits] : contexts)
        any_conflict = any_conflict || (node->getType() == BSDF_TYPE && bits == (ALONE | OVER_BASE));
    if (!any_conflict)
        return 0;

    Splitter splitter(contexts);
    splitter.specialize(root_node, false);
    return splitter.added();
}

} // namespace falcor::mtlx::structured::document_passes
