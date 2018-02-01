# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import logging
import os

import cognitive_face as CF

from egosocial import config
from egosocial.faces.detection import MCSFaceDetector


def list_files(directory, file_pattern='*.txt'):
    '''
    List detection files for social segments.
    :param directory: base directory containing detection files. Must be in the
    social segments tree structure: directory / segm_id / {file}

    :return: list of detection files
    '''
    segments_tree = os.path.join('*', file_pattern)
    return glob.glob(os.path.join(directory, segments_tree))


def get_segment_id(file_path):
    segment_id = os.path.basename(os.path.dirname(file_path))
    return segment_id


class FaceDetectorHelper:

    def __init__(self, detector=None):
        self._detector = MCSFaceDetector() if detector is None else detector
        self._setup_log()

    def _get_files_from_directory(self, directory):
        self._log.debug('Listing files')
        return ((file_path, get_segment_id(file_path)) for file_path in
                list_files(directory, file_pattern='*.jpg'))

    def process_directory(self, input_dir, output_dir):
        self._log.info('Starting face detection')
        # sanity check for input_dir
        self._check_required_directory(input_dir, 'Input')
        # generate output directory if necessary
        self._create_directory(output_dir, 'Output', warn_if_exists=True)

        for image_path, segm_id in self._get_files_from_directory(input_dir):
            # skip segments, TODO: just for debugging
            if segm_id not in ['84']:
                self._log.warning('Skip segment. %s' % segm_id)
                continue

            # generate output directory for a given segment
            segm_output_dir = os.path.join(output_dir, segm_id)
            self._create_directory(segm_output_dir, 'Segment')
            self.process_image(image_path, output_dir, segm_id)

    def process_image(self, image_path, output_dir, segm_id):
        self._log.debug('Processing image %s' % image_path)

        # run detection
        detection_result = self._detect(image_path)
        self._log.debug('Found {} faces'.format(len(detection_result)))

        output_path = self._get_output_path(output_dir=output_dir,
                                            segment_id=segm_id,
                                            input_path=image_path, prefix='')
        # save detection
        self._store(detection_result, output_path)

    def _get_output_path(self, **kwargs):
        image_basename = os.path.basename(kwargs['input_path'])
        # split actual name and extension
        image_name, image_ext = os.path.splitext(image_basename)
        # output path template
        terms = ('{output_dir}', '{s_id}', '{prefix}{im_name}{ext}')
        output_path_tpl = os.path.join(*terms)
        kwargs = dict(output_dir=kwargs['output_dir'],
                      s_id=kwargs['segment_id'], im_name=image_name,
                      ext='.json', prefix=kwargs['prefix'])

        return output_path_tpl.format(**kwargs)

    def _load_input(self, image_path):
        # TODO: MCS API reads image from disk
        return image_path

    def _detect(self, image_path):
        self._log.debug('Running face detection')
        detection_input = self._load_input(image_path)
        faces = self._detector.detect(detection_input)
        return faces

    def _store(self, faces, output_path):
        self._log.debug('Saving face detection %s' % output_path)
        faces_as_dict = [face._asdict() for face in faces]
        with open(output_path, 'w') as json_file:
            json.dump(faces_as_dict, json_file)

    def _setup_log(self):
        self._log = logging.getLogger(__file__)
        self._log.setLevel(logging.DEBUG)
        # create console handler
        ch = logging.StreamHandler()
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to logger
        self._log.addHandler(ch)

    def _check_required_directory(self, directory, description=''):
        error_msg_fmt = '%s directory does not exist %s'
        if not (os.path.exists(directory) and os.path.isdir(directory)):
            error_msg = error_msg_fmt % (description, directory)
            self._log.error(error_msg)
            raise NotADirectoryError(error_msg)

        self._log.debug('%s directory: %s' % (description, directory))

    def _create_directory(self, directory, description='',
                          warn_if_exists=False):
        # generate directory if necessary
        if not (os.path.exists(directory) and os.path.isdir(directory)):
            self._log.debug(
                'Creating %s directory %s' % (description, directory))
            os.makedirs(directory)
        elif warn_if_exists:
            self._log.warning(
                '%s directory already exists %s' % (description, directory))


def main():
    entry_msg = 'Detect faces from social segments.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing social segments.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store to detection results.')

    args = parser.parse_args()

    with open(config.CREDENTIALS_FILE) as credentials_file:
        credentials = json.load(credentials_file)
        KEY = credentials['azure_face_api_key']
        CF.Key.set(KEY)

    helper = FaceDetectorHelper()
    helper.process_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
