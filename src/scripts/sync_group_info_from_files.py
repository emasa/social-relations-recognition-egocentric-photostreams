# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os

from egosocial import config
from egosocial.core.types import FaceClustering, IdentifiedFace
from egosocial.utils.filesystem import check_directory, create_directory
from egosocial.utils.filesystem import list_files_grouped_by_segment, list_segments
from egosocial.utils.logging import setup_logging


class SyncGroupInfoFromFilesHelper:
    """ Helper class to synchronize face groups in social segments.
    """
    def __init__(self, input_dir=None, output_dir=None, groups_file_name=None):
        """

        Args:

        """
        # TODO: add docstring
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._groups_file_name = groups_file_name

        # set up logging
        self._log = logging.getLogger(os.path.basename(__file__))

    def process_segment(self, segment_id):
        # TODO: add docstring
        self._log.info('Synchronizing groups for segment {}'.format(segment_id))

        # check input directory
        segm_input_dir = os.path.join(self._input_dir, segment_id)
        check_directory(segm_input_dir, 'Input Segment')
        # create directory
        segm_output_dir = os.path.join(self._output_dir, segment_id)
        create_directory(segm_output_dir, 'Output Segment', warn_if_exists=True)

        face_clustering = self._get_face_clustering(segm_input_dir)

        output_path = self._get_output_path(output_dir=segm_output_dir,
                                            file_name=self._groups_file_name)
        self._store(face_clustering, output_path)

    def process_all(self):
        # TODO: add docstring
        segments = list_segments(self._input_dir)
        for progress, segment_id in enumerate(segments):
            self.process_segment(segment_id)
            self._log.info('Processed {} / {}'.format(progress+1,
                                                      len(segments)))

    def _get_face_clustering(self, segm_groups_dir):
        # TODO: add docstring
        groups, messyGroup, unknownGroup = [], [], []

        get_name = lambda file_path: os.path.basename(file_path)

        for group, group_id in list_files_grouped_by_segment(
                                    segm_groups_dir,
                                    file_pattern='*.jpg',
                                    output_segment_id=True
                                ):
            if group_id == 'unknownGroup':
                unknownGroup = [IdentifiedFace.create(image_name=get_name(file_path))
                                for file_path in group]
            elif group_id == 'messyGroup':
                messyGroup = [IdentifiedFace.create(image_name=get_name(file_path))
                              for file_path in group]
            else:
                groups.append([IdentifiedFace.create(image_name=get_name(file_path),
                                                     group_id=group_id)
                              for file_path in group])

        clusters = FaceClustering(groups=groups,
                                  messyGroup=messyGroup,
                                  unknownGroup=unknownGroup)

        return clusters

    def _get_output_path(self, **kwargs):
        # TODO: add docstring
        # TODO: precompute output template during init or in static attribute
        # output path template
        terms = ('{output_dir}', '{file_name}')
        output_path_tpl = os.path.join(*terms)
        return output_path_tpl.format(**kwargs)

    def _store(self, face_clustering, output_path):
        # TODO: add docstring
        self._log.debug('Saving groups to %s' % output_path)

        with open(output_path, 'w') as json_file:
            json.dump(face_clustering.to_json(), json_file, indent=4)


def main():
    entry_msg = 'Create heirarchy from social segments by similarity.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing social segments.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store the groups information '
                             'inferred from the input files split.')
    parser.add_argument('--groups_file_name', default='grouped_faces.json',
                        help='File name where to store the groups information.')

    args = parser.parse_args()

    setup_logging(config.LOGGING_CONFIG)

    helper = SyncGroupInfoFromFilesHelper(args.input_dir, args.output_dir,
                                          args.groups_file_name)
    helper.process_all()


if __name__ == '__main__':
    main()