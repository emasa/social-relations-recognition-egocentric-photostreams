# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import shutil

from egosocial import config
from egosocial.core.types import FaceClustering
from egosocial.utils.filesystem import check_directory, create_directory
from egosocial.utils.filesystem import list_files_in_segment, list_segments
from egosocial.utils.logging import setup_logging


class SyncFilesFromGroupInfoHelper:
    """ Helper class to synchronize face groups in social segments.
    """
    def __init__(self, input_dir=None, output_dir=None, groups_dir=None,
                 groups_file_name=None):
        """

        Args:

        """
        # TODO: add docstring
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._groups_dir = groups_dir
        self._groups_file_name = groups_file_name

        # set up logging
        self._log = logging.getLogger(os.path.basename(__file__))

    def process_segment(self, segment_id):
        # TODO: add docstring
        self._log.info('Synchronizing groups for segment {}'.format(segment_id))

        # check input directory
        segm_input_dir = os.path.join(self._input_dir, segment_id)
        check_directory(segm_input_dir, 'Input Segment')
        # check groups directory
        segm_groups_dir = os.path.join(self._groups_dir, segment_id)
        check_directory(segm_groups_dir, 'Groups Segment')
        # load groups and create mapping image_name -> group
        groups_inverse_map = self._load_groups_inverse_map(segm_groups_dir)

        for frame_path in self._segment_sequence(segm_input_dir):
            image_name = os.path.basename(frame_path)

            group_id = self._get_group(image_name, groups_inverse_map)

            # create cluster directory if necessary (also create segment
            # directory the first time)
            cluster_output_dir = os.path.join(self._output_dir, segment_id,
                                              group_id)
            create_directory(cluster_output_dir, 'Cluster', warn_if_exists=True)
            # copy frame from original directory to cluster directory
            output_path = self._get_output_path(output_dir=cluster_output_dir,
                                                image_name=image_name)
            self._store(frame_path, output_path)

    def process_all(self):
        for segment_id in list_segments(self._input_dir):
            self.process_segment(segment_id)

    def _segment_sequence(self, segment_id):
        # TODO: add docstring
        self._log.debug('Listing segment {}'.format(segment_id))
        return list_files_in_segment(self._input_dir, segment_id,
                                     file_pattern='*.jpg')

    def _load_groups_inverse_map(self, segm_groups_dir):
        # TODO: add docstring
        groups_path = os.path.join(segm_groups_dir, self._groups_file_name)

        clusters = self._load_face_clustering(groups_path)

        # TODO: workaround, use iface.group_id (currently it's set to None)
        groups_inverse_map = {self._get_key(iface.image_name) : str(group_id)
                                  for group_id, group in enumerate(clusters.groups)
                                      for iface in group}

        return groups_inverse_map

    def _get_key(self, image_name):
        # TODO: add docstring
        # discard extension to be able to work with different type of files
        # not just the original image
        return os.path.splitext(os.path.basename(image_name))[0]

    def _load_face_clustering(self, groups_path):
        # TODO: add docstring
        self._log.debug('Loading groups %s from ' % groups_path)

        with open(groups_path) as json_file:
            groups_asjson = json.load(json_file)

        return FaceClustering.from_json(groups_asjson)

    def _get_group(self, image_name, groups_inverse_map):
        # TODO: add docstring
        key = self._get_key(image_name)
        if key in groups_inverse_map:
            group_id = groups_inverse_map[key]
        else:
            group_id = 'unknownGroup'

        return group_id

    def _get_output_path(self, **kwargs):
        # TODO: add docstring
        # TODO: precompute output template during init or in static attribute
        # output path template
        terms = ('{output_dir}', '{image_name}')
        output_path_tpl = os.path.join(*terms)
        return output_path_tpl.format(**kwargs)

    def _store(self, frame_path, output_path):
        # TODO: add docstring
        self._log.debug('Copying file from %s to %s' % (frame_path,
                                                        output_path))
        # copy frame from original directory to cluster directory
        shutil.copyfile(frame_path, output_path)


def main():
    entry_msg = 'Create heirarchy from social segments by similarity.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing files using the social '
                             'segments heirarchy.')
    parser.add_argument('--groups_dir', required=True,
                        help='Directory containing the groups information.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store the files using a folder'
                             'for each identified group. Unidentified files '
                             'are stored in a folder called unknownGroup.')
    parser.add_argument('--groups_file_name', default='grouped_faces.json',
                        help='File name containing the groups information.')

    args = parser.parse_args()

    setup_logging(config.LOGGING_CONFIG)

    helper = SyncFilesFromGroupInfoHelper(args.input_dir, args.output_dir,
                                          args.groups_dir, args.groups_file_name)
    helper.process_all()


if __name__ == '__main__':
    main()