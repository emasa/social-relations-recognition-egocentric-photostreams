#!/bin/bash

ITERATIONS=5106
NN_ARCH=VGG
DATA_TYPE=domain

BASE_DIR=/root/shared/Documents/final_proj
MODEL_DIR=$BASE_DIR/models/trained_models/${NN_ARCH}_models/
MODEL_DEF="$MODEL_DIR/end_to_end_training_prototxt/${DATA_TYPE}_eval_net.prototxt"

WEIGHTS_FILE=${DATA_TYPE}_finetune_iter_10000.caffemodel
MODEL_WEIGHTS=$MODEL_DIR/$WEIGHTS_FILE

caffe test -model ${MODEL_DEF} -weights ${MODEL_WEIGHTS} -iterations $ITERATIONS -gpu all