# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The path-traced subsurface walk against a host-side reference on a deep slab.

A slab much thicker than the mean free path and much wider than it is the
semi-infinite half-space the albedo inversion is defined on: a walk entering
through the top face and leaving through it again reflects the authored colour
(Chiang's inversion is fitted so that it does), whatever the radius. The
reference is the same walk run on the host (``_reference_reflectance``), so the
test pins the device walk to the model rather than to a fit tolerance; the
fit's own accuracy is checked separately against its published tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper

# The slab: 2 x 2 wide and 1 deep, top face at z = 0, so a 0.002 mean free path
# puts the sides and bottom out of reach of all but a negligible share of walks.
HALF_WIDTH = 1.0
DEPTH = 1.0
RADIUS = 0.002
THREADS = 1 << 17


def _chiang(albedo: np.ndarray) -> np.ndarray:
    a = np.asarray(albedo, np.float64)
    return 1.0 - np.exp(-5.09406 * a + 2.61188 * a * a - 4.31805 * a**3)


def _reference_reflectance(albedo: float, ior: float = 1.0, paths: int = 200_000, seed: int = 7) -> float:
    """A host walk on a semi-infinite isotropic slab: the share of energy leaving the top.

    Entry is cosine-weighted into the slab. Total internal reflection at ``1/ior``
    is tested on the entry segment only, as the device walk does.
    """
    rng = np.random.default_rng(seed)
    alpha = float(_chiang(albedo))
    sin2_critical = 1.0 / (ior * ior) if ior > 1.0 else 1.0
    z = np.zeros(paths)
    mu = np.sqrt(rng.random(paths))
    weight = np.ones(paths)
    first = np.ones(paths, bool)
    alive = np.ones(paths, bool)
    out = 0.0
    for _ in range(4000):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        d = -np.log(1.0 - rng.random(idx.size))
        z_new = z[idx] + mu[idx] * d
        escaped = z_new < 0.0
        tir = escaped & first[idx] & (1.0 - mu[idx] ** 2 > sin2_critical)
        leave = escaped & ~tir
        out += weight[idx[leave]].sum()
        alive[idx[leave]] = False
        refl = idx[tir]
        z[refl] = 0.0
        mu[refl] = np.sqrt(rng.random(refl.size))
        inside = idx[~escaped]
        z[inside] = z_new[~escaped]
        weight[inside] *= alpha
        mu[inside] = 2.0 * rng.random(inside.size) - 1.0
        first[inside] = False
        low = inside[weight[inside] < 0.2]
        survive = rng.random(low.size) < weight[low] / 0.2
        weight[low[survive]] = 0.2
        alive[low[~survive]] = False
    return out / paths


def _slab_scene(device: spy.Device) -> f2.Scene:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    h, d = HALF_WIDTH, DEPTH
    corners = np.array(
        [[x, y, z] for z in (0.0, -d) for y in (-h, h) for x in (-h, h)], dtype=np.float32
    )
    # Outward-wound quads: top, bottom, and the four sides.
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    positions, indices = [], []
    for quad in quads:
        base = len(positions)
        positions.extend(corners[list(quad)])
        indices.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    positions = np.array(positions, dtype=np.float32)
    normals = np.zeros_like(positions)
    for i in range(0, len(positions), 4):
        n = np.cross(positions[i + 1] - positions[i], positions[i + 2] - positions[i])
        normals[i : i + 4] = n / np.linalg.norm(n)
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=positions,
        sub_mesh_indices=[np.array(indices, dtype=np.uint32)],
        normals=normals,
        name="slab",
    )
    entity = scene.create_entity()
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    scene.update()
    return scene


def _device_reflectance(device: spy.Device, albedo: float, ior: float = 1.0, anisotropy: float = 0.0) -> float:
    scene = _slab_scene(device)
    helper = SceneShaderHelper(device)
    module = helper.get_module(scene, spy.Module.load_from_file(device, "render/test_subsurface_walk.slang"))
    walk = (
        module.subsurface_reflection.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    thread_ids = spy.Tensor.from_numpy(device, np.arange(THREADS, dtype=np.uint32))
    result = walk(
        thread_ids,
        1,
        spy.float3(0.0, 0.0, 0.0),
        spy.float3(0.0, 0.0, 1.0),
        spy.float3(albedo),
        spy.float3(RADIUS),
        anisotropy,
        ior,
    ).to_numpy()
    return float(np.asarray(result).reshape(-1, 3).mean(axis=0).mean())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_single_scatter_albedo_is_chiangs_fit(device_type: spy.DeviceType, device: spy.Device) -> None:
    scene = _slab_scene(device)
    helper = SceneShaderHelper(device)
    module = helper.get_module(scene, spy.Module.load_from_file(device, "render/test_subsurface_walk.slang"))
    for a in (0.05, 0.3, 0.7):
        value = module.single_scatter_albedo(spy.float3(a))
        np.testing.assert_allclose(np.array([value.x, value.y, value.z]), _chiang(a), rtol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("albedo", [0.1, 0.4, 0.8])
def test_slab_reflectance_matches_the_reference_walk(
    device_type: spy.DeviceType, device: spy.Device, albedo: float
) -> None:
    measured = _device_reflectance(device, albedo)
    expected = _reference_reflectance(albedo)
    assert measured == pytest.approx(expected, rel=0.03, abs=0.004)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_slab_reflectance_is_the_authored_albedo(device_type: spy.DeviceType, device: spy.Device) -> None:
    # The inversion's purpose: a semi-infinite slab reflects the authored colour.
    for albedo in (0.2, 0.5):
        assert _device_reflectance(device, albedo) == pytest.approx(albedo, rel=0.06)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_entry_total_internal_reflection_matches_the_reference(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    measured = _device_reflectance(device, 0.4, ior=1.2)
    expected = _reference_reflectance(0.4, ior=1.2)
    assert measured == pytest.approx(expected, rel=0.03, abs=0.004)
