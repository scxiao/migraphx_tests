import numpy as np
#import cv2
import sys
import migraphx

#model = migraphx.parse_onnx("rnn_lstm1layer.onnx")
#names=["dot", "convolution"]
#migraphx.quantize_fp16(model)




#model = migraphx.parse_onnx("..//onnx//alexneti1.onnx")
#names=["dot", "convolution"]
#model.compile(migraphx.get_target("gpu"))
#var = str("program = {}".format(model))
#print("var = {}".format(var))

convert = migraphx.op("dot", **{"alpha": 2.0, "beta": 1.0})
convert = migraphx.op("reduce_mean", **{"axes": [1, 2, 3]})
add = migraphx.op("add")

