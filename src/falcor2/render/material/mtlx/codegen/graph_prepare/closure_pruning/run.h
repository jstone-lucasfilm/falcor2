// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Generated.h>

#include "falcor2/core/macros.h"

#include <cstdint>

MATERIALX_NAMESPACE_BEGIN
class ShaderGraph;
class GenContext;
MATERIALX_NAMESPACE_END

namespace falcor {
namespace mtlx {

class CodegenUserData;

FALCOR_API void
run_closure_pruning(MaterialX::ShaderGraph& graph, MaterialX::GenContext& context, CodegenUserData& user_data);

} // namespace mtlx
} // namespace falcor
