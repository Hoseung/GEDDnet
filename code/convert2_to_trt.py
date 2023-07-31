#import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

params = trt.DEFAULT_TRT_CONVERSION_PARAMS
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='../data/model/V2/',
    conversion_params=params
)

converter.convert()

# Optionally, optimize the model with fake data (especially if using INT8 precision)
# def my_input_fn():
#     yield (input_data, )
# converter.build(input_fn=my_input_fn)

# Save the converted model
converter.save('../data/model/V2_trt/')