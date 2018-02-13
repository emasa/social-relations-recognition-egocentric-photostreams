
#!/bin/bash

# configurable parameters
DATA_TYPE=relation
NN_ARCH=caffeNet
BASE_DIR=/root/shared/Documents/final_proj

WEIGHTS_FILE=finetune_iter_3000.caffemodel
LAYER=fc7
ARCH=GPU

# don't change these parameters unless really neccesary
MODEL_DIR=$BASE_DIR/models/trained_models/${NN_ARCH}_models/
FEATURES_DIR=$BASE_DIR/extracted_features/end_to_end_features/${LAYER}_${DATA_TYPE}_${NN_ARCH}
DUMMY_ATTRIBUTE=${DATA_TYPE}_body

if [ ! -d "$FEATURES_DIR" ]; then
	echo "Creating directory $FEATURES_DIR..."
	mkdir -p $FEATURES_DIR
fi

for SPLIT in "train" "test" "eval";
do
	FEATURES_SPLIT_DIR=$FEATURES_DIR/$SPLIT
	if [ ! -d "$FEATURES_SPLIT_DIR" ]; then
		echo "Creating directory $FEATURES_SPLIT_DIR..."
		mkdir $FEATURES_SPLIT_DIR
	fi

	MODEL_WEIGHTS=$MODEL_DIR/$WEIGHTS_FILE
	if [ -f $MODEL_WEIGHTS ] ; then
		
		MODEL_DEF="$MODEL_DIR/end_to_end_training_prototxt/${DATA_TYPE}_${SPLIT}_net.prototxt"
		OUTPUT_DIR=$FEATURES_SPLIT_DIR/$DUMMY_ATTRIBUTE

		IMAGE_FILE=$(cat $MODEL_DEF | grep -m 1 source | cut -d '"' -f 2)
		N_SAMPLES=$(wc -l < $IMAGE_FILE)

		N_MINI_BATCHES=$N_SAMPLES

		echo "Extracting ${LAYER} from ${DATA_TYPE} data with ${NN_ARCH} for split ${SPLIT}"

		extract_features $MODEL_WEIGHTS $MODEL_DEF $LAYER $OUTPUT_DIR $N_MINI_BATCHES leveldb $ARCH
	else
		echo "Model weights $MODEL_WEIGHTS couldn't be found."
	fi
done