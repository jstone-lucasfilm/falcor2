# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Creases, corners and the triangle subdivision rule shape a refined subdivision surface.

A cube cage refines to a rounded shape. With every edge infinitely sharp it refines to the cube itself, and with
every corner infinitely sharp its corners stay at the cage's points. A semi-sharp crease lies between the two.
"""

import numpy as np
import pytest
import falcor2 as f2
from falcor2.pxr import Usd, UsdGeom

LEVEL = 3
CUBE_POINTS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
CUBE_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]
CUBE_EDGES = sorted({tuple(sorted((f[i], f[(i + 1) % 4]))) for f in CUBE_FACES for i in range(4)})


def _cube(**tags) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    mesh = UsdGeom.Mesh.Define(stage, "/Cube")
    mesh.CreatePointsAttr(CUBE_POINTS)
    mesh.CreateFaceVertexCountsAttr([4] * len(CUBE_FACES))
    mesh.CreateFaceVertexIndicesAttr([i for face in CUBE_FACES for i in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    for name, value in tags.items():
        mesh.GetPrim().GetAttribute(name).Set(value)
    return stage


def _positions(stage: Usd.Stage, level: int = LEVEL) -> np.ndarray:
    options = f2.ImportOptions()
    options.mesh_refinement_level = level
    mesh = f2.UsdImporter().import_stage(stage, options).meshes[0]
    return np.asarray(mesh.positions)


def _every_edge(sharpness: float) -> dict:
    return {
        "creaseIndices": [i for edge in CUBE_EDGES for i in edge],
        "creaseLengths": [2] * len(CUBE_EDGES),
        "creaseSharpnesses": [sharpness] * len(CUBE_EDGES),
    }


def test_an_uncreased_cage_refines_inside_its_hull() -> None:
    positions = _positions(_cube())
    assert np.abs(positions).max() < 0.99


def test_infinitely_sharp_edges_refine_to_the_cage() -> None:
    positions = _positions(_cube(**_every_edge(10.0)))
    # Every refined point lies on a face of the cube: at least one coordinate is at +-1.
    np.testing.assert_allclose(np.abs(positions).max(axis=1), 1.0, atol=1e-5)


def test_a_semi_sharp_crease_lies_between_smooth_and_sharp() -> None:
    extent = [np.abs(_positions(_cube(**_every_edge(s)))).max() for s in (0.0, 1.0, 10.0)]
    assert extent[0] < extent[1] < extent[2]


def test_infinitely_sharp_corners_keep_the_cage_points() -> None:
    positions = _positions(_cube(cornerIndices=list(range(8)), cornerSharpnesses=[10.0] * 8))
    for point in CUBE_POINTS:
        assert np.linalg.norm(positions - np.asarray(point, dtype=np.float32), axis=1).min() < 1e-5


def test_a_chain_reads_as_its_edges() -> None:
    # The cube's bottom ring, authored as one closed chain and as four separate edges.
    ring = [0, 1, 2, 3, 0]
    chain = _positions(_cube(creaseIndices=ring, creaseLengths=[5], creaseSharpnesses=[10.0]))
    per_edge = _positions(_cube(creaseIndices=ring, creaseLengths=[5], creaseSharpnesses=[10.0] * 4))
    edges = _positions(
        _cube(
            creaseIndices=[i for k in range(4) for i in ring[k : k + 2]],
            creaseLengths=[2] * 4,
            creaseSharpnesses=[10.0] * 4,
        )
    )
    np.testing.assert_array_equal(chain, per_edge)
    np.testing.assert_array_equal(chain, edges)
    assert not np.array_equal(chain, _positions(_cube()))


def test_tags_do_not_change_a_cage_refined_at_level_zero() -> None:
    np.testing.assert_array_equal(_positions(_cube(**_every_edge(10.0)), 0), _positions(_cube(), 0))


def test_sharpnesses_matching_neither_creases_nor_edges_are_refused() -> None:
    stage = _cube(creaseIndices=[0, 1, 2], creaseLengths=[3], creaseSharpnesses=[1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="crease sharpnesses"):
        _positions(stage)


def test_the_smooth_triangle_rule_changes_a_triangle_cage() -> None:
    def tetrahedron(rule: str) -> Usd.Stage:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        mesh = UsdGeom.Mesh.Define(stage, "/Tetrahedron")
        mesh.CreatePointsAttr([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])
        mesh.CreateFaceVertexCountsAttr([3] * 4)
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
        mesh.CreateTriangleSubdivisionRuleAttr(rule)
        return stage

    catmull_clark = _positions(tetrahedron(UsdGeom.Tokens.catmullClark))
    smooth = _positions(tetrahedron(UsdGeom.Tokens.smooth))
    assert catmull_clark.shape == smooth.shape
    assert np.abs(catmull_clark - smooth).max() > 1e-3
