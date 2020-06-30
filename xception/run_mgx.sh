#!/bin/bash

exec_loc=/home/scxiao/Workplace/projects/AMDMIGraphX/build/bin
model_loc=/home/scxiao/Workplace/tests/xception
for bs in 1 2 4 8 16 32 48 64
do 
    cmd="${exec_loc}/migraphx-driver perf --batch ${bs} ${model_loc}/xception_opset11.onnx"
    echo $cmd
    $cmd
done

