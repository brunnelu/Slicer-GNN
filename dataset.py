import tensorflow as tf 
import numpy as np 

'''
Loading a training dataset

'''


def print_ds(ds):
    for batch in ds:
        print(batch)
        break
        
def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))

def pack(features, label):
    #because we know all of them have the same label, we take the first one
    return tf.stack(list(features.values()), axis=-1), label[0]

def get_dataset(path, walk_length=20, nr_walks=20,feature_length = 12,train = True):
    
    #ds_files = tf.data.Dataset.list_files(f'{savedir}features/*.tsv')
    column_names= ['y']+ [f'{i}_{j}' for j in range(12) for i in range(walk_length)]
    
    #file names
    if train:
        char = '23456790'
    else:
        char = '8'
    
    ds = tf.data.experimental.make_csv_dataset(
    f'{path}features/[{char}]*.tsv', nr_walks, column_names=column_names,  field_delim='\t', header=False, num_epochs=1, 
        prefetch_buffer_size=None,label_name='y'
    )
    
    #combine all columns to get continous feature and label
    ds = ds.map(pack)
=
    
    for features, labels in ds.take(1):
        print(features.numpy().shape)
        print()
        print(labels.numpy())

    #balance dataset 
    if train:
        ds = ds.filter(lambda x, y: tf.math.logical_or(y >0, tf.random.uniform(shape=[], seed=42)<0.67))
        
    
    # for now we do not care about a good train-val split. This should be fine for the moment
    return ds.shuffle(128).batch(16,drop_remainder=False)
