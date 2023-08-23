import sys
import onnx
import ops
import numpy as np
from tqdm import tqdm
import cv2
import json
import os


def attributes_to_dict(attributes):
    attrs = {}
    for a in attributes:
        typ = onnx.AttributeProto.AttributeType.Name(a.type)
        if typ == 'INTS':
            attrs[a.name] = a.ints
        elif typ == 'INT':
            attrs[a.name] = a.i
        elif typ == 'FLOATS':
            attrs[a.name] = a.floats
        elif typ == 'FLOAT':
            attrs[a.name] = a.f
        else:
            raise NotImplementedError(typ)

    return attrs

model_path = sys.argv[-2]
img_path = sys.argv[-1]

onnx_model = onnx.load(model_path)
tensors = {init.name: onnx.numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

for inp in onnx_model.graph.input:
    if inp.name not in tensors:
        shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        tensors[inp.name] = np.zeros(shape)

# preprocess
im = cv2.imread(img_path)[:,:,::-1].astype(np.float32)
im = cv2.resize(im, (224, 224))
im = np.expand_dims(im, axis=0)
im = np.transpose(im, (0, 3, 1, 2))
im /= 255.0

# set first input
inp_name = onnx_model.graph.node[0].input[0]
tensors[inp_name] = im
print(f'model:\t\t{os.path.split(model_path)[-1]}')
print(f'image path:\t{img_path}')

for layer in tqdm(onnx_model.graph.node):
    try:
        op = getattr(ops, layer.op_type)
    except AttributeError:
        raise NotImplementedError(layer.op_type)

    op(layer.input, layer.output, attributes_to_dict(layer.attribute), tensors)

# get output
with open('./models/imagenet_labels.json', 'r') as f:
    labels = json.load(f)
out = tensors[onnx_model.graph.node[-1].output[0]]
out = np.exp(out) / np.sum(np.exp(out))  # softmax
idxs = np.argsort(-out.reshape(-1))[:5]
for i in idxs:
    print(f'{out[0,i]:.2f}\t{labels[str(i)]}')

