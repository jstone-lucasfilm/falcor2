# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom node libraries: definitions a document instantiates without an XInclude.

A MaterialX application registers its custom nodes once, and every document it
reads may then instantiate them. ``MaterialXMaterial.mtlx_library_paths`` and the
``FALCOR_MTLX_LIBRARY_PATH`` environment variable name such library roots; each
is loaded whole, recursively, after the built-in libraries.
"""

from pathlib import Path

import falcor2 as f2
import falcor2.testing.helpers as helpers
import pytest
import slangpy as spy

# A custom surface defined by a nodegraph over pbrlib, as a studio shading model is.
_DEFINITION = """\
<materialx version="1.39">
  <nodedef name="ND_test_custom_surface" node="test_custom_surface">
    <input name="tint" type="color3" value="0.25, 0.5, 0.75" />
    <output name="out" type="surfaceshader" />
  </nodedef>
  <nodegraph name="NG_test_custom_surface" nodedef="ND_test_custom_surface">
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="color" type="color3" interfacename="tint" />
    </oren_nayar_diffuse_bsdf>
    <surface name="surface" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse" />
    </surface>
    <output name="out" type="surfaceshader" nodename="surface" />
  </nodegraph>
</materialx>
"""

# A material instantiating it, with no XInclude.
_DOCUMENT = """\
<materialx version="1.39">
  <test_custom_surface name="shader" type="surfaceshader">
    <input name="tint" type="color3" value="0.5, 0.25, 0.125" />
  </test_custom_surface>
  <surfacematerial name="material" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="shader" />
  </surfacematerial>
</materialx>
"""


@pytest.fixture
def library(tmp_path: Path) -> Path:
    # Nested one level, to pin that a root is searched recursively.
    root = tmp_path / "library"
    (root / "studio").mkdir(parents=True)
    (root / "studio" / "test_custom_surface.mtlx").write_text(_DEFINITION, encoding="utf-8")
    return root


def _create(device: spy.Device, properties: dict) -> f2.Material:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.MaterialXMaterial,
        f2.Properties({"mtlx_buffer": _DOCUMENT, "mtlx_node_name": "material", **properties}),
    )
    scene.update()
    return material


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_a_library_path_property_defines_the_custom_node(device_type: spy.DeviceType, library: Path) -> None:
    device = helpers.get_device(device_type, use_cache=False)
    try:
        material = _create(device, {"mtlx_library_paths": str(library)})
        assert material.slang_type_name
    finally:
        device.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_the_environment_variable_defines_the_custom_node(
    device_type: spy.DeviceType, library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FALCOR_MTLX_LIBRARY_PATH", str(library))
    device = helpers.get_device(device_type, use_cache=False)
    try:
        material = _create(device, {})
        assert material.slang_type_name
    finally:
        device.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_without_a_library_the_custom_node_is_undefined(
    device_type: spy.DeviceType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FALCOR_MTLX_LIBRARY_PATH", raising=False)
    device = helpers.get_device(device_type, use_cache=False)
    try:
        with pytest.raises(Exception):
            _create(device, {})
    finally:
        device.close()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_a_missing_library_root_is_refused(device_type: spy.DeviceType, tmp_path: Path) -> None:
    device = helpers.get_device(device_type, use_cache=False)
    try:
        with pytest.raises(Exception, match="custom library path"):
            _create(device, {"mtlx_library_paths": str(tmp_path / "missing")})
    finally:
        device.close()


def test_discovery_reads_the_same_libraries(library: Path, tmp_path: Path) -> None:
    path = tmp_path / "material.mtlx"
    path.write_text(_DOCUMENT, encoding="utf-8")
    records = f2.discover_materialx_renderable_elements(path, library_paths=[library])
    assert {record["name"] for record in records} == {"material"}
