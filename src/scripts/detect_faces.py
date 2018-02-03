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
from egosocial.utils.filesystem import list_files
from egosocial.utils.logging import setup_logging


class FaceDetectorHelper:

    def __init__(self, detector=None):
        self._detector = MCSFaceDetector() if detector is None else detector
        self._log = logging.getLogger(os.path.basename(__file__))

    def _get_files_from_directory(self, directory):
        self._log.debug('Listing files')
        return list_files(directory, file_pattern='*.jpg',
                          output_segment_id=True)

    def process_directory(self, input_dir, output_dir):
        self._log.info('Starting face detection')
        # sanity check for input_dir
        check_directory(input_dir, 'Input')
        # generate output directory if necessary
        create_directory(output_dir, 'Output', warn_if_exists=True)

        for image_path, segm_id in self._get_files_from_directory(input_dir):
            # skip segments, TODO: just for debugging
            # if segm_id not in ['84']:
            #    self._log.warning('Skip segment. %s' % segm_id)
            #    continue

            # generate output directory for a given segment
            segm_output_dir = os.path.join(output_dir, segm_id)
            create_directory(segm_output_dir, 'Segment')
            detected_faces = self.process_image(image_path)

            output_path = self._get_output_path(output_dir=segm_output_dir,
                                                input_path=image_path,
                                                prefix='', ext='.json')
            # save detection
            self._store(detected_faces, output_path)

    def process_image(self, image_path):
        self._log.debug('Processing image %s' % image_path)

        detection_input = self._load_input(image_path)
        # run detection
        self._log.debug('Running face detection')
        detection_result = self._detector.detect(detection_input)

        self._log.debug('Found {} faces'.format(len(detection_result)))
        return detection_result

    def _get_output_path(self, **kwargs):
        # image name without extension
        image_name = os.path.splitext(os.path.basename(kwargs['input_path']))[0]
        # output path template
        terms = ('{output_dir}', '{prefix}{image_name}{ext}')
        output_path_tpl = os.path.join(*terms)
        kwargs = dict(output_dir=kwargs['output_dir'], image_name=image_name,
                      ext=kwargs['ext'], prefix=kwargs['prefix'])

        return output_path_tpl.format(**kwargs)

    def _load_input(self, image_path):
        # TODO: MCS API reads image from disk
        return image_path

    def _store(self, faces, output_path):
        self._log.debug('Saving face detection %s' % output_path)
        faces_as_dict = [face._asdict() for face in faces]
        with open(output_path, 'w') as json_file:
            json.dump(faces_as_dict, json_file)


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

    helper = FaceDetectorHelper()
    helper.process_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
