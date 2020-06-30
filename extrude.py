import numpy as np
import trimesh

from global_param import *
from util import vertice2cubevertices


def get_components(mesh):

    componenets = trimesh.graph.connected_components(mesh.edges)
    print([len(x) for x in componenets])

    return componenets


def get_component_faces(mesh):
    # creating graph with
    faces = mesh.faces
    column = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    # column nodes are negative nodes
    column = np.negative(column)
    index = np.tile(np.arange(0, len(faces)), 3)

    edges = np.concatenate(
        (column.reshape((-1, 1)), index.reshape((-1, 1))), axis=1)

    componenets = trimesh.graph.connected_components(edges, engine='networkx')

    componenets = np.array([arr[arr >= 0] for arr in componenets])
    print(componenets)
    print(f'size of each component: {[len(x) for x in componenets]}')
    return componenets


def get_triangle_components(point_components, mesh):
    point_components = set(point_components)
    faces = []

    for i, face in enumerate(mesh.faces):
        if face[0] in point_components and\
            face[1] in point_components and\
                face[2] in point_components:
                    faces.append(i)

    return faces


def exact_savety(faces, mesh):
    meshes = []
    # create 3d triangle for each triangle
    for face in faces:

        vertices = [mesh.vertices[f] for f in mesh.faces[face]]
        vertices = np.array(vertices)

        bounding_vertices = vertice2cubevertices(vertices)
        hull_points = trimesh.convex.hull_points(bounding_vertices)
        # print(hull_points)
        local_mesh = trimesh.convex.convex_hull(hull_points)

        meshes.append(local_mesh)

    local_mesh = trimesh.boolean.union(meshes, 'scad')

    return local_mesh


def approx_savety(faces, mesh):
    # take whole convex hull of bounding cube
    vertices = []
    for face in faces:
        vertices = vertices + ([mesh.vertices[f]
                                   for f in mesh.faces[face]])

    vertices = np.array(vertices)
    vertices = np.unique(vertices, axis=0)
    bounding_vertices = vertice2cubevertices(vertices)
    hull_points = trimesh.convex.hull_points(bounding_vertices)
    # print(hull_points)
    local_mesh = trimesh.convex.convex_hull(hull_points)

    return local_mesh

def componenets2savemesh(component, mesh,source_trimesh):
    # get all triangles in the component
    faces = get_triangle_components(component, mesh)
    # faces = component
    print(f'len faces of component: {len(faces)}')
    # only keep unique faces
    faces = list(set(faces))
    print(f'len faces of component: {len(faces)}')
    

    if(len(faces)<=exact_limit):
        local_mesh = exact_savety(faces,mesh)
    else:
        local_mesh = approx_savety(faces,mesh)

    return local_mesh

def extrude(mesh, source_trimesh):
    
    print(f'source mesh is waterthight: {source_trimesh.is_watertight}')
    assert(source_trimesh.is_watertight)
    
    # get separate components
    components = get_components(mesh)#get_component_faces(mesh)#
    # components = [components[0],components[2],components[3]]
    #components = [components[1]]
    connected_components = []
    
    # actual extrusion
    for component in components:
        processed_component = componenets2savemesh(component, mesh,source_trimesh)
        print(processed_component.is_watertight)
        connected_components.append(processed_component)

    components = connected_components

    is_watertight = [x.is_watertight for x in components]
    print(f'Are all extruded components waterthight: {np.all(is_watertight)}')

    
    processed_component = trimesh.boolean.union(components,'scad')

    critical_mesh = processed_component.intersection(source_trimesh,'scad')
    processed_source_mesh = source_trimesh.difference(processed_component,'scad')
    
    print(f'Are final meshes waterthight:{processed_source_mesh.is_watertight}(save) {critical_mesh.is_watertight}(critical)')

    return critical_mesh, processed_source_mesh
