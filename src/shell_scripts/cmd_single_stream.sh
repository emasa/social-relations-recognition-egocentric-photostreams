
#!/bin/bash

ARCH=GPU
LAYER=fc7
NN_ARCH=caffeNet
DATA_TYPE=relation

BASE_DIR=$HOME/shared/Documents/final_proj
DEF_DIR=$BASE_DIR/models/trained_models/${NN_ARCH}_models/end_to_end_training_prototxt
TEMPLATE_MODEL_DEF="$DEF_DIR/template_single_stream_net.prototxt"
TMP_MODEL_DEF="$DEF_DIR/tmp_single_stream_net.prototxt"
BASE_SPLIT_DIR=$BASE_DIR/datasets/splits
SPLITS_DIR=$BASE_SPLIT_DIR/annotator_consistency3

case "$DATA_TYPE" in
	*domain*) N_CLASSES=5 ; DOM_REL_PREFIX="domain_single" ;;
	*relation*) N_CLASSES=16 ; DOM_REL_PREFIX="single" ;;
esac

ATTRIBUTES_DIR=$BASE_DIR/models/trained_models/attribute_models
FEATURES_DIR=$BASE_DIR/extracted_features/attribute_features/${LAYER}_${DATA_TYPE}_${NN_ARCH}

echo "Extracting feature ${LAYER} from ${DATA_TYPE} data..."

if [ ! -d "$FEATURES_DIR" ]; then
	echo "Creating directory $FEATURES_DIR..."
	mkdir -p $FEATURES_DIR
fi

## now loop through the above array
for SPLIT in "train" "test" "eval";
do
	FEATURES_SPLIT_DIR=$FEATURES_DIR/$SPLIT
	if [ ! -d "$FEATURES_SPLIT_DIR" ]; then
		echo "Creating directory $FEATURES_SPLIT_DIR..."
		mkdir $FEATURES_SPLIT_DIR
	fi

	for MODEL_DIR in $(find $ATTRIBUTES_DIR -mindepth 1 -maxdepth 1 -type d)
	do	
		MODEL_WEIGHTS=$MODEL_DIR/finetune_iter_1000.caffemodel
		if [ -f $MODEL_WEIGHTS ] ; then
			ATTRIBUTE=$(basename $MODEL_DIR)

			case "$ATTRIBUTE" in
    			*body*) IMAGE_TYPE="body" ;;
				*face*|*head*) IMAGE_TYPE="face" ;;
			esac

			for ID in 1 2;
			do
				IMAGE_FILE="$SPLITS_DIR/${DOM_REL_PREFIX}_${IMAGE_TYPE}${ID}_${SPLIT}_${N_CLASSES}.txt"

				cat $TEMPLATE_MODEL_DEF | sed "s#source: \"#\0$IMAGE_FILE#" > $TMP_MODEL_DEF

				OUTPUT_DIR="$FEATURES_SPLIT_DIR/${ATTRIBUTE}_${ID}"
				N_SAMPLES=$(wc -l < $IMAGE_FILE)
				N_MINI_BATCHES=$N_SAMPLES

				echo "Processing attribute ${ATTRIBUTE}_${ID} for split ${SPLIT} with ${DATA_TYPE} data"
				
				extract_features $MODEL_WEIGHTS $TMP_MODEL_DEF $LAYER $OUTPUT_DIR $N_MINI_BATCHES leveldb $ARCH

				echo "Deleting temporary files..."
				rm $TMP_MODEL_DEF
			done
		fi
	done
done