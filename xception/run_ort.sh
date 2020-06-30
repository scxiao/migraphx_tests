#!/bin/bash

for bs in 1 2 4 8 16 32 48 64
do 
    cmd="python3 run_xception.py xception_opset11.onnx --batch_size ${bs}"
    echo $cmd
    $cmd
done
