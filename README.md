# FeatureID

This project is a proof of concept on how ML can be used to find critical parts for AM. Our new approach allows to easily separate input meshes in critical and non critical parts. For training artificial dataa is created.

## Requirements
pip3 install trimesh open3d networkx scipy pandas numpy

apt get install openscad (on Debian based systems)

## Use it

`data_gen.py` generates artificial trainingdata

`train.py` trains a model on the generated data

`val.py` can be used to visualize how the model performs on validation meshes

`split.py` splits input meshes into critical and non critical meshes depending on which faces are critical