# tensorflow2timeloop

Unofficial converter of Tensorflow Keras models in [Timeloop](https://github.com/NVlabs/timeloop) workload yaml files. Inspired by [pythorch2timeloop](https://github.com/Accelergy-Project/pytorch2timeloop-converter).  
Current supported layers:
* `tf.keras.layers.Conv2D`
* `tf.keras.layers.DepthwiseConv2D`
* `tf.keras.layers.Dense`

### Installation
After cloning this repository, run `python setup.py install` to finish the installation.

### Usage
```python
import tensorflow as tf
import tensorflow2timeloop

# Define a Keras neural network model, for example, a pre-defined MobileNet from tensorflow applications.
model = tf.keras.applications.MobileNet()

# Define the number of batches that will be used for the inference
batch_size = 1

# Define the directory names where the timeloop workload yaml files will be stored.
# The yaml files will be stored in ./workloads/mobilenet/ in this example.
top_dir = 'workloads'
sub_dir = 'mobilenet'

# Convert
tensorflow2timeloop.convert_model(model, batch_size, sub_dir, top_dir)
```
Layers with the same workloads bounds will be converted only once because their timeloop simulation would result in the same outcome. 
A `layerbook.txt` is generated in order to map the actual layers of the model to the converted ones.  
Each line is in the form: `keras_layer_name, yaml_file_index`. In the MobileNet example:
```
conv1, 1
conv_dw_1, 2
conv_pw_1, 3
conv_dw_2, 4
conv_pw_2, 5
conv_dw_3, 6
conv_pw_3, 7
conv_dw_4, 8
conv_pw_4, 9
conv_dw_5, 10
conv_pw_5, 11
conv_dw_6, 12
conv_pw_6, 13
conv_dw_7, 14
conv_pw_7, 15
conv_dw_8, 14
conv_pw_8, 15
conv_dw_9, 14
conv_pw_9, 15
...
```
