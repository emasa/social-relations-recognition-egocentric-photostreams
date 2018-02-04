# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os

import cognitive_face as cf

from egosocial import config
from egosocial.core.types import Face, IdentifiedFace
from egosocial.faces.clustering import MCSFaceClustering
from egosocial.utils.filesystem import check_directory, create_directory
from egosocial.utils.filesystem import list_files_in_segment, list_segments
from egosocial.utils.logging import setup_logging


class FaceGroupHelper:
    """ Helper class to group faces in social segments from filesystem.
    """

    def __init__(self, input_dir=None, output_dir=None, groups_file_name=None,
                 face_clustering=None):
        """

        Args:
            :param input_dir: directory containing social segment folders.
            :param output_dir: directory where to store face groups. It creates
            the social segments folder structure.
            :param face_clustering: face clustering method (callback).
        """
        # TODO: fix docstring
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._groups_file_name = groups_file_name
        # use Microsoft Cognitive Services by default.
        self._face_clustering = (MCSFaceClustering() if not face_clustering
                                                     else face_clustering)
        # set up logging
        self._log = logging.getLogger(os.path.basename(__file__))

    def process_segment(self, segment_id):
        """ Group faces in social segments.

        Args:
            :param segment_id: segment id.
        """
        self._log.info('Face clustering for segment {}'.format(segment_id))

        # sanity check for input_dir
        check_directory(self._input_dir, 'Input')
        # create directory
        segm_output_dir = os.path.join(self._output_dir, segment_id)
        create_directory(segm_output_dir, 'Output Segment', warn_if_exists=True)
        # load detected faces and create face groups
        ifaces = self._load_ifaces_from_segment(segment_id)
        clusters = self._face_clustering(ifaces)

        # save results
        output_path = self._get_output_path(output_dir=segm_output_dir,
                                            file_name=self._groups_file_name)
        self._store(clusters, output_path)

    def process_all(self):
        # TODO: add docstring
        for segment_id in list_segments(self._input_dir):
            self.process_segment(segment_id)

    def _get_detections(self, segment_id):
        """ list face detection files in given segment.

        Args:
            :param segment_id: segment id. Detection files are expected in
            json format.
        Returns:
            :return: list of face detection files.
        """
        self._log.debug('Listing detection files in segment {}'.format(segment_id))
        return list_files_in_segment(self._input_dir, segment_id,
                                     file_pattern='*.json')

    def _load_ifaces_from_segment(self, segment_id):
        """ Load faces from a set of detection files.
        Args:
            :param segment_id: segment id. Detection files are expected in
            json format.
        Returns:
            :return: identified faces.
        """
        return [IdentifiedFace.create(image_name=face.image_name,
                                      face_id=face.face_id)
                for file_path in self._get_detections(segment_id)
                    for face in self._load_input(file_path)
               ]

    def _load_input(self, detection_path):
        """ Load faces from detection file.

        Args:
            :param detection_path: path to detection file.
        Returns:
            :return: faces.
        """
        self._log.debug('Loading detection file %s' % detection_path)
        with open(detection_path) as json_file:
            face_detection = json.load(json_file)

        return (Face.from_json(face_asdict) for face_asdict in face_detection)

    def _get_output_path(self, **kwargs):
        # TODO: add docstring
        # TODO: precompute output template during init or in static attribute
        # output path template
        terms = ('{output_dir}', '{file_name}')
        output_path_tpl = os.path.join(*terms)
        return output_path_tpl.format(**kwargs)

    def _store(self, clusters, output_path):
        # TODO: add docstring
        self._log.debug('Storing face groups in %s' % output_path)

        with open(output_path, 'w') as json_file:
            json.dump(clusters.to_json(), json_file, indent=4)


def main():
    entry_msg = 'Group faces from social segments by similarity.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing social segments.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store the grouped results.')
    parser.add_argument('--groups_file_name', default='grouped_faces.json',
                        help='File name where to store the groups information.')

    args = parser.parse_args()

    setup_logging(config.LOGGING_CONFIG)

    with open(config.CREDENTIALS_FILE) as credentials_file:
        credentials = json.load(credentials_file)
        cf.Key.set(credentials['azure_face_api_key'])
        cf.BaseUrl.set(config.FACE_API_BASE_URL)

    helper = FaceGroupHelper(args.input_dir, args.output_dir,
                             args.groups_file_name)
    helper.process_all()


if __name__ == '__main__':
    main()
