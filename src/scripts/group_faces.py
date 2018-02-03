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
from egosocial.utils.filesystem import list_files_grouped_by_segment
from egosocial.utils.logging import setup_logging


class FaceGroupHelper:
    """ Helper class to group faces in social segments from filesystem.
    """

    def __init__(self, face_clustering=None):
        """

        Args:
            :param face_clustering: face clustering method (callback).
        """
        # use Microsoft Cognitive Services by default.
        self._face_clustering = MCSFaceClustering() if face_clustering is None \
            else face_clustering
        # set up logging
        self._log = logging.getLogger(os.path.basename(__file__))

    def process_directory(self, input_dir, output_dir):
        """ Group faces in social segments.

        Args:
            :param input_dir: directory containing social segment folders.
            :param output_dir: directory where to store face groups. It creates
            the social segments folder structure.
        """
        self._log.info('Starting face clustering')
        # sanity check for input_dir
        check_directory(input_dir, 'Input')
        # generate output directory if necessary
        create_directory(output_dir, 'Output', warn_if_exists=True)

        for detection_files, segm_id in self._get_detections(input_dir):
            # skip segments, TODO: just for debugging
            # if segm_id not in ['84']:
            #    self._log.warning('Skip segment. %s' % segm_id)
            #    continue
            ifaces = self._load_ifaces_from_segment(detection_files)
            clustering_result = self._process_batch(ifaces)

            # create directory
            segm_output_dir = os.path.join(output_dir, segm_id)
            create_directory(segm_output_dir, 'Segment', warn_if_exists=True)
            # save results
            output_path = self._get_output_path(output_dir=segm_output_dir,
                                                name='grouped_faces',
                                                ext='.json')
            self._store(clustering_result, output_path)

    def _get_detections(self, directory):
        """ for each segment list detected faces.

        Args:
            :param input_dir: directory containing social segment folders.
            Detection files are expected in json format.
        Returns:
            :return: lazy iterator of list of detected faces grouped by segment.
        """
        self._log.debug('Listing detection files')
        return list_files_grouped_by_segment(directory, file_pattern='*.json',
                                             output_segment_id=True)

    def _load_ifaces_from_segment(self, detection_files):
        """ Load faces from a set of detection files.
        Args:
            :param detection_files: list of detection file paths.
        Returns:
            :return: identified faces.
        """
        return [
            IdentifiedFace(bbox=face.bbox, image_name=face.image_name,
                           face_id=face.face_id, group_id=None)
            for file_path in detection_files
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

        return [Face(**face_asdict) for face_asdict in face_detection]

    def _process_batch(self, ifaces):
        # TODO: add docstring
        return self._face_clustering(ifaces)

    def _get_output_path(self, **kwargs):
        # TODO: add docstring
        # TODO: precompute output template during init or in static attribute
        # output path template
        terms = ('{output_dir}', '{name}{ext}')
        output_path_tpl = os.path.join(*terms)
        return output_path_tpl.format(**kwargs)

    def _store(self, clusters, output_path):
        # TODO: add docstring
        self._log.debug('Storing face groups in %s' % output_path)

        clustering_dict = {
            'groups': [[iface._asdict() for iface in group]
                       for group in clusters[0]],
            'messyGroup': [iface._asdict() for iface in clusters[1]],
            'unknownGroup': [iface._asdict() for iface in clusters[2]]
        }

        with open(output_path, 'w') as json_file:
            json.dump(clustering_dict, json_file)


def main():
    entry_msg = 'Group faces from social segments by similarity.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing social segments.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store the grouped results.')

    args = parser.parse_args()

    setup_logging(config.LOGGING_CONFIG)

    with open(config.CREDENTIALS_FILE) as credentials_file:
        credentials = json.load(credentials_file)
        cf.Key.set(credentials['azure_face_api_key'])
        cf.BaseUrl.set(config.FACE_API_BASE_URL)

    helper = FaceGroupHelper()
    helper.process_directory(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
