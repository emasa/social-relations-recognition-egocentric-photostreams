
#!/bin/bash

BASE_DIR=/root/shared/Documents/final_proj/datasets/social_segments
INPUT_DIR=$BASE_DIR/detected_images
OUTPUT_DIR=$BASE_DIR/compressed_detected_images

OLD_PWD=$PWD

echo "Entering $INPUT_DIR..."
cd $INPUT_DIR

for SEGM_DIR in $(ls -d */*/)
do
	INPUT_IMAGE_DIR=$INPUT_DIR/$SEGM_DIR
	OUT_IMAGE_DIR=$OUTPUT_DIR/$SEGM_DIR
	
	if [ ! -d "$OUT_IMAGE_DIR" ]; then
		echo "Creating folder $OUT_IMAGE_DIR..."
		mkdir -p $OUT_IMAGE_DIR
	fi

	echo "Entering $OUT_IMAGE_DIR..."
	cd $INPUT_IMAGE_DIR
	# Copying files
	# for IMAGE_FILE in $(ls *.txt | cut -d . -f 1)
	# do
	# 	cp $IMAGE_FILE.txt $OUT_IMAGE_DIR/$IMAGE_FILE.txt
	# 	echo "Copying $IMAGE_FILE.txt..." 
	# done
	# Compressing	
	for IMAGE_FILE in $(ls *.txt | cut -d . -f 1)
	do
		cp $IMAGE_FILE.png $OUT_IMAGE_DIR/$IMAGE_FILE.jpg
		echo "Compressing $IMAGE_FILE with jpg format..." 
	done	
done

echo "Leaving $PWD..."
cd $OLD_PWD