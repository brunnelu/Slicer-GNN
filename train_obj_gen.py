import open3d as o3d
import numpy as np
import copy
import trimesh
import math
from critical import get_critical, get_critical_normal_array, get_critical_object_array
from util import draw_registration_result, o3d2trimesh, trimesh_mesh2o3d_mesh, draw_critical, draw_critical_prob
from scipy.spatial.transform import Rotation as R

import random
"""  Generation of artificial training meshes. """

def center_mesh(mesh):
    mesh.translate(-mesh.get_center())
    return mesh


def subtract(mesh, mesh_sub, union=False):
    mesh = o3d2trimesh(mesh)
    mesh_sub = o3d2trimesh(mesh_sub)
    if not union:
        res = mesh.difference(mesh_sub, 'scad')
    else:
        res = trimesh.boolean.union([mesh, mesh_sub], 'scad')

    return trimesh_mesh2o3d_mesh(res)


def union(meshes):
    meshes = [o3d2trimesh(mesh) for mesh in meshes]
    res = trimesh.boolean.union(meshes, 'scad')
    return trimesh_mesh2o3d_mesh(res)


def gen_vertical_hole(diam=0.7, critical_=0.7, thickness=2, dimension=20, edges_cylinder=64, draw=False):
    # critical dependent of print settings

    box = o3d.geometry.TriangleMesh.create_box(dimension, dimension, thickness)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        diam, thickness, edges_cylinder)

    box = center_mesh(box)
    cylinder = center_mesh(cylinder)

    res = subtract(box, cylinder)

    critical = get_critical_object_array(box, cylinder, res)

    if diam > critical_:
        critical = np.zeros(critical.shape)

    if draw:
        draw_critical(res, critical)

    return res, critical


def gen_horizontal_hole(diam=0.77, critical=0.7,thickness=2, dimension=20, edges_cylinder=64, draw=False):
    # diam >7mm critical
    res, critical = gen_vertical_hole(diam, critical, thickness=thickness,
                                      dimension=dimension, edges_cylinder=edges_cylinder, draw=False)
    res = res.rotate(R.from_rotvec([np.pi/2, 0, 0]).as_matrix(),center=(0,0,0))

    if draw:
        draw_critical(res, critical)

    return res, critical

def gen_overhanging_side(angle=45, length=10, critical_bound=(30, 45), thickness=3, draw=False):
    # overhanging and critical if angle in bound

    angle = np.deg2rad(angle)

    #torus and shere
    dimension = length
    pole = o3d.geometry.TriangleMesh.create_box(
        thickness, dimension, thickness)
    side = o3d.geometry.TriangleMesh.create_box(
        thickness, dimension, thickness)

    # center boxes
    pole = center_mesh(pole)
    side = center_mesh(side)

    # random translation
    side.rotate(R.from_rotvec([0, 0, -angle]).as_matrix(),center=(0,0,0))
    side.translate([thickness+0.1, 2, 0])
    res = union([pole, side])

    res = res.rotate(R.from_rotvec([np.pi/2, 0, 0]).as_matrix(),center=(0,0,0))

    critical = get_critical_normal_array(res, critical_bound=critical_bound)

    if draw:
        draw_critical(res, critical)

    return res, critical


def gen_overhang_double(angle=45, length=10, critical_bound=(30, 45), thickness=3, draw=False):
    # roof
    # roof consist of two overhanging sides`
    # apparently there is some numerically instability and we sould relax the angle for the right overhanging side
    angle = 90-angle+1e-4
    left, _ = gen_overhanging_side(
        angle=angle, length=length, critical_bound=critical_bound, thickness=thickness)
    right, _ = gen_overhanging_side(
        angle=angle, length=length, critical_bound=critical_bound, thickness=thickness)

    left.rotate(R.from_rotvec([-np.pi/2, 0, 0]).as_matrix(),center=(0,0,0))
    right.rotate(R.from_rotvec([-np.pi/2, 0, 0]).as_matrix(),center=(0,0,0))

    # rotate right to oposite side

    right.rotate(R.from_rotvec([0, np.pi, 0]).as_matrix(),center=(0,0,0))
    left.translate([-length/math.sqrt(2), 0, 0])

    # random translation
    res = union([left, right])
    res = center_mesh(res)
    res = res.rotate(R.from_rotvec([np.pi/2, 0, 0]).as_matrix(),center=(0,0,0))
    critical = get_critical_normal_array(res, critical_bound=critical_bound)

    if draw:
        draw_critical(res, critical)

    return res, critical


def gen_fine_wall_single(thickness=0.5, critical_=0.5, dimension=8, ext=2, out=True, draw=False):
    # wall or thing gap is critical
    box = o3d.geometry.TriangleMesh.create_box(
        dimension, dimension, 2*dimension)
    box2 = o3d.geometry.TriangleMesh.create_box(
        dimension, dimension, thickness)

    box = center_mesh(box)
    box2 = center_mesh(box2)
    box2 = box2.translate([0, ext, 1])

    if out:
        res = subtract(box, box2)
    else:
        res = subtract(box, box2, union=True)
    critical = get_critical_object_array(box, box2, res)

    if critical_ < thickness:
        critical = np.zeros(critical.shape)

    res = res.rotate(R.from_rotvec([np.pi/2, 0, 0]).as_matrix(),center=(0,0,0))

    if draw:
        draw_critical(res, critical)

    return res, critical


def gen_fine_walls(thickness=0.5, critical_=0.5, dimension=5, ext=2, nr_boxes=10, translate_factor=4, draw=False):
    # thin walls
    create_box = (lambda: o3d.geometry.TriangleMesh.create_box(
        dimension, 3, thickness))
    boxes = []

    for _ in range(nr_boxes):
        boxes.append(create_box().rotate(R.from_rotvec([0, random.random() * np.pi, 0]).as_matrix(),center=(0,0,0)))

    # center boxes
    boxes = [center_mesh(box) for box in boxes]

    # random translation
    boxes = [box.translate([translate_factor*random.random(),
                            0, translate_factor*random.random()]) for box in boxes]

    res = union(boxes)

    res = res.rotate(R.from_rotvec([np.pi/2, 0, 0]).as_matrix(),center=(0,0,0))

    critical = get_critical_normal_array(res, critical_bound=(90-1e-4, 180))

    if critical_ < thickness:
        critical = np.zeros(critical.shape)

    if draw:
        draw_critical(res, critical)

    return res, critical



if __name__ == "__main__":
    gen_horizontal_hole(diam=0.3, critical=0.7,thickness=2, dimension=20, edges_cylinder=64, draw=True)
    
    exit()
    mesh = gen_fine_walls(draw=True)
    mesh = gen_overhang_double(draw=True)
    mesh, array = gen_overhanging_side(draw=True)
    mesh = gen_vertical_hole(draw=True)
    mesh = gen_horizontal_hole(draw=True)
    mesh = gen_fine_wall_single(draw=True)
    mesh = gen_fine_wall_single(out=False, draw=True)
