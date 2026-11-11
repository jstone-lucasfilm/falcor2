# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closure pruning's asymmetric mix fold, more than once in one graph.

A mix whose one side is provably empty (a zero-weight BSDF) folds to
``multiply(live, amount)``. The fold creates its helper nodes after MaterialX
has named every variable in the graph, so each helper must be named itself:
left with its bare port name, two folds in one graph both declared ``BSDF out``
and the generated module failed to compile. Each case compiles with pruning on
and evaluates and reports exactly as the same document with pruning off.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers

_LEAVES = """
  <oren_nayar_diffuse_bsdf name="base" type="BSDF">
    <input name="color" type="color3" value="0.5, 0.45, 0.4" />
  </oren_nayar_diffuse_bsdf>
  <oren_nayar_diffuse_bsdf name="stain" type="BSDF">
    <input name="color" type="color3" value="0.3, 0.2, 0.1" />
  </oren_nayar_diffuse_bsdf>
  <oren_nayar_diffuse_bsdf name="dust" type="BSDF">
    <input name="color" type="color3" value="0.7, 0.7, 0.6" />
  </oren_nayar_diffuse_bsdf>
  <dielectric_bsdf name="stain_empty" type="BSDF">
    <input name="weight" type="float" value="0" />
    <input name="scatter_mode" type="string" value="T" />
  </dielectric_bsdf>
  <dielectric_bsdf name="dust_empty" type="BSDF">
    <input name="weight" type="float" value="0" />
    <input name="scatter_mode" type="string" value="T" />
  </dielectric_bsdf>
  <texcoord name="uv" type="vector2" />
  <extract name="uv_u" type="float">
    <input name="in" type="vector2" nodename="uv" />
    <input name="index" type="integer" value="0" />
  </extract>"""


def _amount(runtime: bool, value: float) -> str:
    # The runtime form reads the texture coordinate, which the probe sets to 0.5,
    # so no static analysis can know its value.
    return 'nodename="uv_u"' if runtime else f'value="{value}"'


def _stain_layer(runtime: bool) -> str:
    return f"""
  <mix name="stain_mix" type="BSDF">
    <input name="fg" type="BSDF" nodename="stain" />
    <input name="bg" type="BSDF" nodename="stain_empty" />
    <input name="mix" type="float" {_amount(runtime, 0.5)} />
  </mix>
  <layer name="stained" type="BSDF">
    <input name="top" type="BSDF" nodename="stain_mix" />
    <input name="base" type="BSDF" nodename="base" />
  </layer>"""


def _dust_layer(runtime: bool) -> str:
    return f"""
  <mix name="dust_mix" type="BSDF">
    <input name="fg" type="BSDF" nodename="dust" />
    <input name="bg" type="BSDF" nodename="dust_empty" />
    <input name="mix" type="float" {_amount(runtime, 0.5)} />
  </mix>
  <layer name="dusted" type="BSDF">
    <input name="top" type="BSDF" nodename="dust_mix" />
    <input name="base" type="BSDF" nodename="stained" />
  </layer>"""


def _document(folds: int, runtime: bool) -> str:
    body = _stain_layer(runtime) + (_dust_layer(runtime) if folds == 2 else "")
    root = "dusted" if folds == 2 else "stained"
    return f"""<materialx version="1.39" colorspace="lin_rec709">{_LEAVES}{body}
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="{root}" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""


_LAYERING_MODES = [
    f2.MaterialXLayeringMode.closure_tree,
    f2.MaterialXLayeringMode.bsdf_mix,
]

_DEVICE_TYPES = [
    device_type
    for device_type in helpers.DEFAULT_DEVICE_TYPES
    if device_type in (spy.DeviceType.d3d12, spy.DeviceType.vulkan)
]


def _probe(
    device: spy.Device,
    document: str,
    layering_mode: f2.MaterialXLayeringMode,
    optimize: f2.MaterialXOptimizeGraphFlags,
) -> list[dict]:
    scene = f2.Scene.create(device)
    props = f2.Properties()
    props["mtlx_buffer"] = document
    props["mtlx_node_name"] = "M"
    props["mtlx_layering_mode"] = layering_mode
    props["mtlx_optimize_graph"] = optimize
    material = scene.create_material(f2.MaterialXMaterial, props)
    scene.update()

    from falcor2.editor import SceneShaderHelper

    helper = SceneShaderHelper(device)
    # The shared-input probe reads what this test needs: leaf weights and the BSDF.
    source_module = spy.Module.load_from_file(device, "render/test_materialx_shared_bsdf_input.slang")
    module, _ = helper.prepare(scene, source_module)
    probe = (
        module.shared_bsdf_input_probe.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    wi = np.array([(0.0, 1.0, 0.0), (0.6, 0.8, 0.0)], dtype=np.float32)
    wo = np.array([(0.0, 1.0, 0.0), (-0.28, 0.96, 0.0)], dtype=np.float32)
    results = probe(
        seeds=np.array([(7, 0, 7), (8, 0, 7)], dtype=np.uint32),
        material_id=int(material.material_id),
        uv=np.full((2, 2), 0.5, dtype=np.float32),
        wi=wi,
        wo=wo,
    )
    cursor = results.cursor()
    cursor.load()
    return [cursor[i].read() for i in range(2)]


def _eval(result: dict) -> np.ndarray:
    value = result["eval"]
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float64)


def _weights(result: dict) -> np.ndarray:
    # Sorted, since pruning removes the empty leaves and the two reports need not
    # list the live ones at the same indices.
    count = int(result["bsdf_count"])
    live = [float(result["bsdf_weight"][i].x) for i in range(count)]
    return np.sort([w for w in live if w > 0.0])


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize("layering_mode", _LAYERING_MODES)
@pytest.mark.parametrize("runtime", [False, True], ids=["constant_amount", "runtime_amount"])
@pytest.mark.parametrize("folds", [1, 2], ids=["one_fold", "two_folds"])
def test_folded_mixes_compile_and_match_the_unpruned_graph(
    device_type: spy.DeviceType, layering_mode: f2.MaterialXLayeringMode, runtime: bool, folds: int
) -> None:
    device = helpers.get_device(device_type)
    document = _document(folds, runtime)
    pruned = _probe(device, document, layering_mode, f2.MaterialXOptimizeGraphFlags.closure_pruning)
    unpruned = _probe(device, document, layering_mode, f2.MaterialXOptimizeGraphFlags.none)

    for got, expected in zip(pruned, unpruned):
        np.testing.assert_allclose(_eval(got), _eval(expected), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(_weights(pruned[0]), _weights(unpruned[0]), rtol=1e-5)
