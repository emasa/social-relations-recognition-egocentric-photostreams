#!/bin/bash

NN_ARCH=mobileNet
DATA_TYPE=domain

BASE_DIR=/root/shared/Documents/final_proj
MODEL_DIR=$BASE_DIR/models/trained_models/${NN_ARCH}_models/
SOLVER_DEF="$MODEL_DIR/end_to_end_training_prototxt/${DATA_TYPE}_solver.prototxt"

#WEIGHTS_FILE=domain_finetune_iter_3000.caffemodel
WEIGHTS_FILE=mobilenet.caffemodel
MODEL_WEIGHTS=$MODEL_DIR/$WEIGHTS_FILE

caffe train -solver ${SOLVER_DEF} -weights ${MODEL_WEIGHTS}