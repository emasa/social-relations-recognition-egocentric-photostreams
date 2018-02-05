# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os

import cognitive_face as CF

from egosocial import config
from egosocial.faces.detection import MCSFaceDetector
from egosocial.utils.filesystem import check_directory, create_directory
from egosocial.utils.filesystem import list_files_in_segment, list_segments
from egosocial.utils.logging import setup_logging


class FaceDetectorHelper:

    def __init__(self, input_dir=None, output_dir=None, detector=None):
        # TODO: fix docstring
        self._input_dir = input_dir
        self._output_dir = output_dir
        # use Microsoft Cognitive Services by default.
        self._face_detection = MCSFaceDetector() if not detector else detector

        self._log = logging.getLogger(os.path.basename(__file__))

    def process_segment(self, segment_id):
        # TODO: add docstring
        self._log.info('Face detection for segment {}'.format(segment_id))
        # sanity check for input_dir
        check_directory(self._input_dir, 'Input')
        # create directory
        segm_output_dir = os.path.join(self._output_dir, segment_id)
        create_directory(segm_output_dir, 'Output Segment', warn_if_exists=True)

        for image_path in self._get_images(segment_id):
            detected_faces = self.process_image(image_path)

            output_path = self._get_output_path(output_dir=segm_output_dir,
                                                input_path=image_path,
                                                ext='.json')
            # save detection
            self._store(detected_faces, output_path)

    def process_all(self):
        # TODO: add docstring
        segments = list_segments(self._input_dir)
        for progress, segment_id in enumerate(segments):
            self.process_segment(segment_id)
            self._log.info('Processed {} / {}'.format(progress+1,
                                                      len(segments)))

    def _get_images(self, segment_id):
        # TODO: add docstring
        self._log.debug('Listing images in segment {}'.format(segment_id))
        return list_files_in_segment(self._input_dir, segment_id,
                                     file_pattern='*.jpg')

    def process_image(self, image_path):
        # TODO: add docstring
        self._log.debug('Processing image %s' % image_path)
        detection_input = self._load_input(image_path)
        # run detection
        self._log.debug('Running face detection')
        detection_result = self._face_detection(detection_input)

        self._log.debug('Found {} faces'.format(len(detection_result)))
        return detection_result

    def _get_output_path(self, **kwargs):
        # image name without extension
        image_name = os.path.splitext(os.path.basename(kwargs['input_path']))[0]
        # output path template
        terms = ('{output_dir}', '{image_name}{ext}')
        output_path_tpl = os.path.join(*terms)
        kwargs = dict(output_dir=kwargs['output_dir'], image_name=image_name,
                      ext=kwargs['ext'])

        return output_path_tpl.format(**kwargs)

    def _load_input(self, image_path):
        # TODO: MCS API reads image from disk
        return image_path

    def _store(self, faces, output_path):
        self._log.debug('Saving face detection %s' % output_path)
        faces_json = [face.to_json() for face in faces]
        with open(output_path, 'w') as json_file:
            json.dump(faces_json, json_file, indent=4)


def main():
    entry_msg = 'Detect faces from social segments.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing social segments.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store to detection results.')

    args = parser.parse_args()
    setup_logging(config.LOGGING_CONFIG)

    with open(config.CREDENTIALS_FILE) as credentials_file:
        credentials = json.load(credentials_file)
        CF.Key.set(credentials['azure_face_api_key'])
        CF.BaseUrl.set(config.FACE_API_BASE_URL)

    helper = FaceDetectorHelper(args.input_dir, args.output_dir)
    helper.process_all()


if __name__ == '__main__':
    main()
