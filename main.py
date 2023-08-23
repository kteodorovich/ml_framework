import sys
import onnx
import ops
import numpy as np
from tqdm import tqdm


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


onnx_model = onnx.load(sys.argv[-1])

tensors = {}

tensors = {init.name: onnx.numpy_helper.to_array(
    init) for init in onnx_model.graph.initializer}

for inp in onnx_model.graph.input:
    if inp.name not in tensors:
        shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        tensors[inp.name] = np.zeros(shape)

# set first input
inp_name = onnx_model.graph.node[0].input[0]
tensors[inp_name] = np.zeros(tensors[inp_name].shape)

for layer in tqdm(onnx_model.graph.node):
# for layer in onnx_model.graph.node:
    try:
        # print(layer.name)
        op = getattr(ops, layer.op_type)
    except AttributeError:
        raise NotImplementedError(layer.op_type)

    op(layer.input, layer.output, attributes_to_dict(layer.attribute), tensors)

# get output
out = tensors[onnx_model.graph.node[-1].output[0]]
print(out.shape)
idxs = np.argsort(-out.reshape(-1))[:5]
print(idxs.shape)
for i in idxs:
    print(f'{i}:\t{out[0,i]}')

