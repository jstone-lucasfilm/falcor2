# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A BSDF that reaches a mix twice keeps its weight on both paths.

``mix(layer(top, base), base, amount)`` reads ``base`` on both sides of the mix,
the shape of a coverage-weighted layer such as ``LamaLayer``. Inside a nodegraph
the mix's ``amount`` is an interface input, so it is connected, and MaterialX's
``premultipliedBsdfAdd`` refactor would lower the mix to
``add(multiply(fg, amount), multiply(bg, 1 - amount))``. A BSDF ``multiply``
scales its operand's weight in place, so the ``1 - amount`` meant for the bg
path also scaled the base inside the layer, and two nested instances drove the
base to zero. Each case is a nodegraph instance compared against the same graph
written inline, which the refactor never lowered: the two must evaluate and
report alike in both layering modes.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers

_COVER_DEFINITION = """
  <nodedef name="ND_cover_twice" node="cover_twice" nodegroup="pbr">
    <input name="top" type="BSDF" />
    <input name="base" type="BSDF" />
    <input name="amount" type="float" value="1.0" />
    <output name="out" type="BSDF" />
  </nodedef>
  <nodegraph name="NG_cover_twice" nodedef="ND_cover_twice">
    <layer name="layered" type="BSDF">
      <input name="top" type="BSDF" interfacename="top" />
      <input name="base" type="BSDF" interfacename="base" />
    </layer>
    <mix name="coverage" type="BSDF">
      <input name="fg" type="BSDF" nodename="layered" />
      <input name="bg" type="BSDF" interfacename="base" />
      <input name="mix" type="float" interfacename="amount" />
    </mix>
    <output name="out" type="BSDF" nodename="coverage" />
  </nodegraph>"""

_LEAVES = """
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
    <input name="color" type="color3" value="0.5, 0.45, 0.4" />
  </oren_nayar_diffuse_bsdf>
  <dielectric_bsdf name="coat" type="BSDF">
    <input name="ior" type="float" value="1.54" />
    <input name="roughness" type="vector2" value="0.0625, 0.0625" />
    <input name="scatter_mode" type="string" value="R" />
  </dielectric_bsdf>
  <dielectric_bsdf name="water" type="BSDF">
    <input name="ior" type="float" value="1.334" />
    <input name="roughness" type="vector2" value="0.0025, 0.0025" />
    <input name="scatter_mode" type="string" value="R" />
  </dielectric_bsdf>"""


def _instance(name: str, top: str, base: str, amount: float) -> str:
    return f"""
  <cover_twice name="{name}" type="BSDF">
    <input name="top" type="BSDF" nodename="{top}" />
    <input name="base" type="BSDF" nodename="{base}" />
    <input name="amount" type="float" value="{amount}" />
  </cover_twice>"""


def _inline(name: str, top: str, base: str, amount: float) -> str:
    return f"""
  <layer name="{name}_layered" type="BSDF">
    <input name="top" type="BSDF" nodename="{top}" />
    <input name="base" type="BSDF" nodename="{base}" />
  </layer>
  <mix name="{name}" type="BSDF">
    <input name="fg" type="BSDF" nodename="{name}_layered" />
    <input name="bg" type="BSDF" nodename="{base}" />
    <input name="mix" type="float" value="{amount}" />
  </mix>"""


def _document(body: str) -> str:
    return f"""<materialx version="1.39" colorspace="lin_rec709">{_COVER_DEFINITION}{_LEAVES}{body}
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="root" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""


# (instance body, inline body, the diffuse base's expected weight at normal incidence).
# One cover at 0.5 leaves the base fully seen on the uncovered half and seen
# through the coat on the covered half; nesting the coat at 1 under water at 0.5
# is the teeth's shape, where the defect zeroed the base outright.
_CASES = {
    "one_cover_half": (
        _instance("root", "coat", "diffuse", 0.5),
        _inline("root", "coat", "diffuse", 0.5),
    ),
    "nested_outer_half": (
        _instance("inner", "coat", "diffuse", 1.0) + _instance("root", "water", "inner", 0.5),
        _inline("inner", "coat", "diffuse", 1.0) + _inline("root", "water", "inner", 0.5),
    ),
    "nested_inner_half": (
        _instance("inner", "coat", "diffuse", 0.5) + _instance("root", "water", "inner", 1.0),
        _inline("inner", "coat", "diffuse", 0.5) + _inline("root", "water", "inner", 1.0),
    ),
}

_LAYERING_MODES = [
    f2.MaterialXLayeringMode.closure_tree,
    f2.MaterialXLayeringMode.bsdf_mix,
]

_DEVICE_TYPES = [
    device_type
    for device_type in helpers.DEFAULT_DEVICE_TYPES
    if device_type in (spy.DeviceType.d3d12, spy.DeviceType.vulkan)
]

# The diffuse leaf's kind, k_bsdf_kind_oren_nayar.
_DIFFUSE_KIND = 3


def _probe(device: spy.Device, document: str, layering_mode: f2.MaterialXLayeringMode) -> dict:
    scene = f2.Scene.create(device)
    props = f2.Properties()
    props["mtlx_buffer"] = document
    props["mtlx_node_name"] = "M"
    props["mtlx_layering_mode"] = layering_mode
    material = scene.create_material(f2.MaterialXMaterial, props)
    scene.update()

    from falcor2.editor import SceneShaderHelper

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_materialx_shared_bsdf_input.slang")
    module, _ = helper.prepare(scene, source_module)
    probe = (
        module.shared_bsdf_input_probe.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    # Normal incidence and a second, oblique direction pair.
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


def _diffuse_weight(result: dict) -> float:
    for index in range(int(result["bsdf_count"])):
        if round(float(result["bsdf_kind"][index])) == _DIFFUSE_KIND:
            return float(result["bsdf_weight"][index].x)
    raise AssertionError("no diffuse leaf reported")


def _eval(result: dict) -> np.ndarray:
    value = result["eval"]
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float64)


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize("layering_mode", _LAYERING_MODES)
@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_bsdf_read_twice_through_a_nodegraph_matches_the_inline_graph(
    device_type: spy.DeviceType, layering_mode: f2.MaterialXLayeringMode, case: str
) -> None:
    device = helpers.get_device(device_type)
    instance_body, inline_body = _CASES[case]
    instance = _probe(device, _document(instance_body), layering_mode)
    inline = _probe(device, _document(inline_body), layering_mode)

    for got, expected in zip(instance, inline):
        np.testing.assert_allclose(_eval(got), _eval(expected), rtol=1e-5, atol=1e-7)
    assert _diffuse_weight(instance[0]) == pytest.approx(_diffuse_weight(inline[0]), rel=1e-5)
    # The base is seen on both the covered and the uncovered paths, so it never
    # reports zero; the defect drove it to exactly that on the nested shape.
    assert _diffuse_weight(instance[0]) > 0.5
