import numpy as np
#import cv2
import sys
import migraphx

#model = migraphx.parse_onnx("rnn_lstm1layer.onnx")
model = migraphx.parse_onnx("..//onnx//alexneti1.onnx")
#names=["dot", "convolution"]
names=["dot", "convolution"]
#migraphx.quantize_fp16(model)
model.compile(migraphx.get_target("gpu"))

