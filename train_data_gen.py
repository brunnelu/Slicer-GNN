import numpy as np
import pandas as pd
import os
import open3d as o3d
import networkx as nx
from random import uniform as u
from random import randint as r
from random import gauss as g
from util import randow_walk, draw_critical_prob
from train_obj_gen import gen_fine_wall_single, gen_fine_walls, gen_horizontal_hole, gen_vertical_hole, gen_overhanging_side, gen_overhang_double
from dual import get_dual

savedir = '/home/lucas/Documents/data/'


def gen_sample(id_, walk_length=20, nr_walks=20, draw=True):

    sample = "overhanging_side"
    choise = np.random.choice(6, p=[0.2, 0.2, 0.2, 0.15, 0.2, 0.05])
    print(choise)

    switch = {
        0: (lambda: gen_vertical_hole(diam=u(0.2,1.5), critical_=0.7, thickness=u(1,5), dimension=u(10,25), edges_cylinder=r(12,124), draw=draw), "vertical_hole"),
        1: (lambda: gen_horizontal_hole(diam=u(0.2,1.5), critical=0.7, thickness=u(1,5), dimension=u(10,25), edges_cylinder=r(12,124), draw=draw), "horizontal_hole"),
        2: (lambda: gen_overhanging_side(angle=r(10,70), length=10, critical_bound=(30, 45), thickness=u(2,4), draw=draw), "overhanging_side"),
        3: (lambda: gen_fine_wall_single(thickness=u(0.2,1.5), critical_=0.5, dimension=u(5,10), ext=u(1,3), out=True, draw=draw), "fine_wall_single_out"),
        4: (lambda: gen_fine_wall_single(thickness=u(0.2,1.), critical_=0.5, dimension=u(5,10), ext=u(1,3), out=False, draw=draw), "fine_wall_single_in"),
        5: (lambda: gen_fine_walls(thickness=u(0.2,1.5), critical_=0.5, dimension=u(3,10), ext=u(0.5,4), nr_boxes=r(2,12), translate_factor=u(2,6), draw=draw), "fine_walls")
    }

    func, sample = switch.get(choise)

    mesh, critical = func()
    G, face_feature = get_dual(mesh)

    #if all faces are critical something went wrong and we discard this sample
    if np.all(critical):
        print('discard sample')
        return None

    df = []

    for start in range(len(critical)):
        label = critical[start]
        for _ in range(nr_walks):
            walk, feature = randow_walk(start, G, walk_length, face_feature)
            row = np.concatenate((np.array([label]), feature))
            df.append(row.reshape((1,-1)))


    df = pd.DataFrame(np.concatenate(df,axis=0))

    df.to_csv(savedir+f'features/{id_}.tsv',header=False ,sep='\t', index=False)

    o3d.io.write_triangle_mesh(savedir+f'stl/{id_}.stl', mesh)

    return sample


def create_samples(n=10, walk_length=20, nr_walks=20, draw=True):
    # load samples
    df = pd.read_csv(savedir+f'samples.tsv', sep='\t')

    # create new samples
    for _ in range(n):
        id_ = len(df)+1
        sample = gen_sample(id_, walk_length=walk_length, nr_walks=nr_walks,draw=draw)

        if not sample is None:
            df = df.append({'id': id_, 'sample': sample}, ignore_index=True)

    # save id of samples
    df.to_csv(savedir+f'samples.tsv',
              header=["id", "sample"], sep='\t', index=False)


if __name__ == "__main__":
    
    create_samples(n=1000,draw=False)

    exit()
    mesh = gen_fine_walls(draw=True)
    mesh, array = gen_overhang_double(draw=True)

    mesh = gen_vertical_hole(draw=True)
    mesh = gen_horizontal_hole(draw=True)
    mesh = gen_fine_wall_single(draw=True)
    mesh = gen_fine_wall_single(out=False, draw=True)

    exit()
    #fix sample
    df = pd.DataFrame()
    df = df.append({'id': 1, 'sample': 'bla'}, ignore_index=True)
    df.to_csv(savedir+f'samples.tsv',
              header=["id", "sample"], sep='\t', index=False)