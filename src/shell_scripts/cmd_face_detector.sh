
#!/bin/bash

BASE_DIR=/root/shared/Documents/final_proj/datasets/social_segments
INPUT_DIR=$BASE_DIR/images
OUTPUT_DIR=$BASE_DIR/detected_images
DETECTOR_FOLDER=/opt/py-faster-rcnn
THRESHOLD=0.85

OLD_PWD=$PWD

echo "Entering $INPUT_DIR..."
cd $INPUT_DIR

for SEGM_DIR in $(ls -d */*/)
do
	case "$SEGM_DIR" in
		*train*) IN_IMAGE_DIR=$INPUT_DIR/$SEGM_DIR;;
		*test*)  IN_IMAGE_DIR=$INPUT_DIR/$SEGM_DIR/data;;
	esac
	OUT_IMAGE_DIR=$OUTPUT_DIR/$SEGM_DIR
	
	if [ ! -d "$OUT_IMAGE_DIR" ]; then
		echo "Creating folder $OUT_IMAGE_DIR..."
		mkdir -p $OUT_IMAGE_DIR
	fi

	if [ "$DETECTOR_FOLDER" != "$PWD" ]; then
		echo "Entering $DETECTOR_FOLDER..."
		cd $DETECTOR_FOLDER
	fi

	python $DETECTOR_FOLDER/tools/facedetection_images.py --gpu 0 --image_folder $IN_IMAGE_DIR --output_folder $OUT_IMAGE_DIR --conf_thresh $THRESHOLD
done

echo "Leaving $PWD..."
cd $OLD_PWD