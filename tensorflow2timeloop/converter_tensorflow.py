import os
import tensorflow.keras.layers as layers
from .yaml_templates import convolution_template, depthwise_convolution_template 

def generate_Conv2D(layer, batch_size, unique_layers, layers_info):
    assert layer.groups == 1, "Groups not supported yet."

    dims = {}
    dims['c'] = layer.input_shape[3]
    dims['m'] = layer.filters
    dims['n'] = batch_size
    dims['p'] = layer.output_shape[1]
    dims['q'] = layer.output_shape[2]
    dims['r'] = layer.kernel_size[0]
    dims['s'] = layer.kernel_size[1]
    dims['wd'] = layer.dilation_rate[0]
    dims['hd'] = layer.dilation_rate[1]
    dims['ws'] = layer.strides[0]
    dims['hs'] = layer.strides[1]

    dims_frozen = frozenset(dims.items())

    if dims_frozen not in unique_layers:
        unique_layers.append(dims_frozen)
        layers_info.append([layer.name, len(unique_layers)])
        return convolution_template.format(**dims)
    else:
        layers_info.append([layer.name, unique_layers.index(dims_frozen)+1])
        return False

def generate_DepthwiseConv2D(layer, batch_size, unique_layers, layers_info):
    assert layer.depth_multiplier == 1, "Depth multiplier not supported yet."

    dims = {}
    dims['c'] = layer.input_shape[3]
    dims['n'] = batch_size
    dims['p'] = layer.output_shape[1]
    dims['q'] = layer.output_shape[2]
    dims['r'] = layer.kernel_size[0]
    dims['s'] = layer.kernel_size[1]
    dims['wd'] = layer.dilation_rate[0]
    dims['hd'] = layer.dilation_rate[1]
    dims['ws'] = layer.strides[0]
    dims['hs'] = layer.strides[1]

    dims_frozen = frozenset(dims.items())

    if dims_frozen not in unique_layers:
        unique_layers.append(dims_frozen)
        layers_info.append([layer.name, len(unique_layers)])
        return depthwise_convolution_template.format(**dims)
    else:
        layers_info.append([layer.name, unique_layers.index(dims_frozen)+1])
        return False

def generate_Dense(layer, batch_size, unique_layers, layers_info):
    dims = {}
    dims['c'] = layer.input_shape[-1]
    dims['m'] = layer.output_shape[-1]
    dims['n'] = batch_size
    dims['p'] = 1
    dims['q'] = 1
    dims['r'] = 1
    dims['s'] = 1
    dims['wd'] = 1
    dims['hd'] = 1
    dims['ws'] = 1
    dims['hs'] = 1

    dims_frozen = frozenset(dims.items())

    if dims_frozen not in unique_layers:
        unique_layers.append(dims_frozen)
        layers_info.append([layer.name, len(unique_layers)])
        return convolution_template.format(**dims)
    else:
        layers_info.append([layer.name, unique_layers.index(dims_frozen)+1])
        return False

def convert_model(model, batch_size, model_name, save_dir):
    outdir = os.path.join(save_dir, model_name)
    os.makedirs(outdir, exist_ok=True)

    unique_layers = []
    layers_info = []
    layer_index = 1

    for layer in model.layers:
        yaml = False
        if isinstance(layer, layers.DepthwiseConv2D):
            yaml = generate_DepthwiseConv2D(layer, batch_size, unique_layers, layers_info)
        elif isinstance(layer, layers.Conv2D):
            yaml = generate_Conv2D(layer, batch_size, unique_layers, layers_info)
        elif isinstance(layer, layers.Dense):
            yaml = generate_Dense(layer, batch_size, unique_layers, layers_info)

        if yaml:
            file_name = model_name + '_' + 'layer' + str(layer_index) + '.yaml'
            file_path = os.path.abspath(os.path.join(save_dir, model_name, file_name))
            with open(file_path, "w") as f:
                f.write(yaml)

            layer_index += 1

    file_name = model_name + '_layerbook.txt'
    file_path = os.path.abspath(os.path.join(save_dir, model_name, file_name))
    with open(file_path, "w") as f:
        for layer in layers_info:
            f.write("{}, {}\n".format(layer[0], layer[1]))
