#!/usr/bin/env sh

TOOLS=./build/tools

#$TOOLS/caffe train \
  #--solver=ldj_workspace/caltech101/solver.prototxt -gpu 0

#./build/tools/caffe train -solver ldj_workspace/caltech101/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0

./build/tools/caffe train -solver ldj_workspace/caltech101/squzz_solver.prototxt -weights models/squeezenet_v1.0.caffemodel -gpu 0

# 首次fine-tuing的初始化值 models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu 0

#models/finetune_flickr_style/finetune_flickr_style.caffemodel -gpu 0
