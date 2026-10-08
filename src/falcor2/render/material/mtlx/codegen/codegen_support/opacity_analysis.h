// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Generated.h>

#include "../graph_prepare/static_input_query/static_input_query.h"

#include <optional>

MATERIALX_NAMESPACE_BEGIN
class ShaderGraph;
MATERIALX_NAMESPACE_END

namespace falcor {
namespace mtlx {
namespace codegen_support {

// Returns the exact root surface opacity after traversing expanded MaterialX
// implementation graphs, or {} when any required value is not statically known.
std::optional<float>
materialx_graph_static_opacity(const MaterialX::ShaderGraph& graph, const StaticInputQuery& static_input_query);

} // namespace codegen_support
} // namespace mtlx
} // namespace falcor
