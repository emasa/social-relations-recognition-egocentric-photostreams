# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import logging
import os
from collections import namedtuple

import cognitive_face as CF

from egosocial import config
from egosocial.core.types import Face

IdentifiedFace = namedtuple('IdenfiedFace',
                            ('segment_id', 'image_name', 'face_id', 'group_id'))


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


class FaceGroupHelper:

    def __init__(self):
        self._setup_log()

    def _get_files_from_directory(self, directory):
        self._log.debug('Listing files')
        return ((file_path, get_segment_id(file_path)) for file_path in
                list_files(directory, file_pattern='*.json'))

    def process_directory(self, input_dir, output_dir):
        self._log.info('Starting face detection')
        # sanity check for input_dir
        self._check_required_directory(input_dir, 'Input')
        # generate output directory if necessary
        self._create_directory(output_dir, 'Output', warn_if_exists=True)

        all_id_faces = []
        for file_path, segm_id in self._get_files_from_directory(input_dir):
            # skip segments, TODO: just for debugging
            # if segm_id not in ['84']:
            #    self._log.warning('Skip segment. %s' % segm_id)
            #    continue

            image_name = os.path.splitext(os.path.basename(file_path))[
                             0] + '.jpg'
            faces = self._load_input(file_path)
            id_faces = [
                IdentifiedFace(segm_id, image_name, face.params['faceId'], None)
                for face in faces]

            all_id_faces.extend(id_faces)

        self.process_batch(all_id_faces, output_dir)

    def process_batch(self, id_faces, output_dir):
        self._log.debug('Process batch of images')

        # run _group_faces
        faces_by_similarity = self._group_faces(id_faces)
        self._log.debug('Found {} groups'.format(len(faces_by_similarity)))

        output_path = self._get_output_path(output_dir=output_dir,
                                            prefix='group')
        # save results
        self._store(faces_by_similarity, output_path)

    def _get_output_path(self, **kwargs):
        # output path template
        terms = ('{output_dir}', '{prefix}{ext}')
        output_path_tpl = os.path.join(*terms)
        kwargs = dict(output_dir=kwargs['output_dir'], prefix=kwargs['prefix'],
                      ext='.json')
        return output_path_tpl.format(**kwargs)

    def _load_input(self, detection_path):
        self._log.debug('Loading detection file %s' % detection_path)
        with open(detection_path) as json_file:
            face_detection = json.load(json_file)

        faces = [Face(**face_asdict) for face_asdict in face_detection]

        return faces

    def _group_faces(self, id_faces):
        self._log.debug('Running face grouping')
        result = CF.face.group([iface.face_id for iface in id_faces])
        face_id_map = {iface.face_id: iface for iface in id_faces}

        def set_group_id(iface, group_id):
            return IdentifiedFace(*iface[:-1], group_id=group_id)

        groups = [
            [set_group_id(face_id_map[face_id], group_id) for face_id in group]
            for group_id, group in enumerate(result['groups'])]
        groups.append(
            [set_group_id(face_id_map[face_id], 'messyGroup') for face_id in
             result['messyGroup']])

        return groups

    def _store(self, groups, output_path):
        self._log.debug('Saving face groups %s' % output_path)
        ifaces_as_dict = [[iface._asdict() for iface in group] for group in
                          groups]
        with open(output_path, 'w') as json_file:
            json.dump(ifaces_as_dict, json_file)

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
    entry_msg = 'Group faces from social segments by similarity.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing social segments.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store the grouped results.')

    args = parser.parse_args()

    with open(config.CREDENTIALS_FILE) as credentials_file:
        credentials = json.load(credentials_file)
        KEY = credentials['azure_face_api_key_2']
        CF.Key.set(KEY)

        DEFAULT_BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com' \
                           '/face/v1.0/'
        CF.BaseUrl.set(DEFAULT_BASE_URL)

    helper = FaceGroupHelper()
    helper.process_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
