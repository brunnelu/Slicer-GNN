'''
Copyright Lucas Brunner 2020

Training a model on our artificial training dataset
'''

import numpy as np 
import pandas as pd 
import tensorflow as tf
import pandas as pd
import os

from dataset import get_dataset

import argparse



savedir = './data/'


print(f'dataset path: {savedir}')


def print_ds(ds):
    count = 0
    all_samples = 0
    for feature, label in ds:
        count += sum(label.numpy())
        all_samples += len(label.numpy())
        
    print(f'num samples: {all_samples}')
    print(f'num critical: {count}')

def get_model(walk_length=20, nr_walks=20,feature_length = 12):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(nr_walks,walk_length*feature_length)))
    model.add(tf.keras.layers.Reshape((nr_walks,walk_length,feature_length)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(16,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(32,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(16,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(8,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(4,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(4,3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(4,3, activation='relu'))
    model.add(tf.keras.layers.Flatten()) 
    
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))
    
    model.summary()
    
    return model

def get_model4(walk_length=20, nr_walks=20,feature_length = 12):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(nr_walks,walk_length*feature_length)))
    model.add(tf.keras.layers.Reshape((nr_walks,walk_length,feature_length)))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(8 , activation='relu'))

    model.add(tf.keras.layers.Reshape((nr_walks,walk_length*8)))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    
    model.add(tf.keras.layers.Flatten())

    
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(8 , activation='relu'))
    model.add(tf.keras.layers.Dense(1 , activation='relu'))
    
    model.summary()
    
    return model


def train():
    parser = argparse.ArgumentParser(description='Which model to run')
    parser.add_argument('m', default=0, type=int, nargs='?', help='model number')
    args = parser.parse_args()

    print(f'running model {args.m}')
    if args.m == 4:
        model = get_model4(walk_length=20, nr_walks=20,feature_length = 12)
    else:
        model = get_model(walk_length=20, nr_walks=20,feature_length = 12)

    ds_train = get_dataset(path=savedir,walk_length=20, nr_walks=20,feature_length = 12,train=True)
    ds_val = get_dataset(path=savedir,walk_length=20, nr_walks=20,feature_length = 12,train=False)
    
    model.compile(
    loss=[tf.keras.losses.MeanSquaredError(name='mse')],

    optimizer='adam',
    metrics=['accuracy','mse'])
    
    hist = model.fit(x=ds_train,epochs=100,validation_data=ds_val,verbose=2)

    model.save("my_model")


if __name__ == "__main__":
    train()