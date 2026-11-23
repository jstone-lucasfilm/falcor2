// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Document.h>

namespace falcor::mtlx::structured::document_passes {

/// Give each layering context of a shared BSDF its own node.
///
/// A BSDF whose output feeds two consumers may sit in the top of a layer over
/// another BSDF in one of them and stand on its own in the other: a glass that
/// is both a mix's branch and the coat of a layer beside it. The layering
/// analysis decides per closure whether a transmissive lobe's transmission
/// leaves the material or continues to the layers beneath, and one closure
/// cannot hold both answers, so the glass would be compiled as a coat in both
/// places. A closure is a value, so a copy evaluates as the original does;
/// copying the shared BSDF, and the BSDFs upstream of it, for the context that
/// layers it over a base lets each use be classified on its own.
///
/// Only BSDFs reached in both contexts are copied, and pattern inputs stay
/// shared, so a document without such a closure is left unchanged. The pass
/// works on the graph that holds `root`; implementation nodegraphs are read,
/// never edited. Returns the number of nodes added.
std::size_t
split_shared_closures_by_layering_context(MaterialX::Document& document, const MaterialX::TypedElementPtr& root);

} // namespace falcor::mtlx::structured::document_passes
