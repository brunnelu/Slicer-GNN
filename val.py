import numpy as np
import pandas as pd
import os
import open3d as o3d
import networkx as nx
import copy

from global_param import *
from util import o3d2trimesh, trimesh_mesh2o3d_mesh, draw_registration_result, fix_input, randow_walk, draw_critical_prob, draw_critical
from critical import get_critical
from extrude import extrude
from dual import get_dual

from random import uniform as u
from random import randint as r
from random import gauss as g
from train_obj_gen import center_mesh
from dual import get_dual


from tensorflow import keras

""" Validating a trained model by predicting how critical the faces are """


walk_length=20
nr_walks=20

def validate(path, draw_prob = True):
    source = o3d.io.read_triangle_mesh(path)
    
    # preprocessing, to position and size as training data
    # the scale is dependent on the resolution of your training data
    source.scale(0.25, center=source.get_center())
    source = center_mesh(source)
    print(source)

    #compute vertex normals for nicer looks
    #triangle normals are submitted by the stl file
    source.compute_vertex_normals()

    #give form base colour
    source.paint_uniform_color(blue)

    #fix to get waterthight object
    mesh = fix_input(source)

    G, face_feature = get_dual(mesh)

    #o3d.visualization.draw_geometries([mesh])

    model = keras.models.load_model('my_model2')
    model.summary()
    
    print(face_feature.shape)

    
    labels = []
    for start in range(len(face_feature)):
        df = []
        for _ in range(nr_walks):
            walk, feature = randow_walk(start, G, walk_length, face_feature)
            df.append(feature.reshape((1,20,12)))

        df = np.concatenate(df,axis=0).reshape((1,20,-1))
        label = model.predict(df)
        labels.append(label)

    labels = np.array(labels).reshape(-1)
    
    print(labels)

    if draw_prob:
        draw_critical_prob(source, labels)

    # binarise for visual
    cutoff = 0.1
    labels[labels>cutoff] = 1.
    labels[labels<=cutoff] = 0.

    #print(labels)
    draw_critical(source, labels)


if __name__ == "__main__":
    validate("../model/Demo_1.stl")