# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os

from egosocial import config
from egosocial.core.types import FaceClustering
from egosocial.utils.filesystem import check_directory, create_directory
from egosocial.utils.filesystem import list_files_in_segment, list_segments
from egosocial.utils.logging import setup_logging


class CreateFacePairsHelper:
    """ Create face pairs (appearing in the same image) in social segments.
    """
    def __init__(self, input_dir=None, output_dir=None,
                 groups_file_name=None, pairs_file_name=None):
        """

        Args:

        """
        # TODO: add docstring
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._groups_file_name = groups_file_name
        self._pairs_file_name = pairs_file_name

        # set up logging
        self._log = logging.getLogger(os.path.basename(__file__))

    def process_segment(self, segment_id):
        # TODO: add docstring
        self._log.info('Create face pairs for segment {}'.format(segment_id))

        # check input directory
        segm_input_dir = os.path.join(self._input_dir, segment_id)
        check_directory(segm_input_dir, 'Input Segment')
        # create output directory
        segm_output_dir = os.path.join(self._output_dir, segment_id)
        create_directory(segm_output_dir, 'Output Segment', warn_if_exists=True)

        iface_pairs = self.get_face_pairs(segment_id)

        # copy frame from original directory to cluster directory
        output_path = self._get_output_path(output_dir=segm_output_dir,
                                            file_name=self._pairs_file_name)

        self._store(iface_pairs, output_path)

    def get_face_pairs(self, segment_id):
        # TODO: add docstring
        from itertools import combinations
        from operator import itemgetter, attrgetter
        # create mapping source image-> [identified faces]
        grouped_by_source = self._load_ifaces_grouped_by_source(segment_id)
        # sort by source
        sorted_items = sorted(grouped_by_source.items(), key=itemgetter(0))

        iface_pairs = []

        for src, ifaces in sorted_items:
            # sort by group_id
           sorted_ifaces = sorted(ifaces, key=attrgetter('image_name'))
           source_pairs = combinations(sorted_ifaces, 2)
           iface_pairs.extend(source_pairs)

        return iface_pairs

    def process_all(self):
        # TODO: add docstring
        segments = list_segments(self._input_dir)
        for progress, segment_id in enumerate(segments):
            self.process_segment(segment_id)
            self._log.info('Processed {} / {}'.format(progress+1,
                                                      len(segments)))

    def _segment_sequence(self, segment_id):
        # TODO: add docstring
        self._log.debug('Listing segment {}'.format(segment_id))
        return list_files_in_segment(self._input_dir, segment_id,
                                     file_pattern='*.json')

    def _load_ifaces_grouped_by_source(self, segment_id):
        # TODO: add docstring
        groups_path = os.path.join(self._input_dir, segment_id,
                                   self._groups_file_name)

        clusters = self._load_face_clustering(groups_path)

        import collections
        grouped_by_image = collections.defaultdict(list)

        for group in clusters.groups:
            for iface in group:
                grouped_by_image[self._get_key(iface.image_name)].append(iface)

        return grouped_by_image

    def _load_face_clustering(self, groups_path):
        # TODO: add docstring
        self._log.debug('Loading groups %s from ' % groups_path)

        with open(groups_path) as json_file:
            groups_asjson = json.load(json_file)

        return FaceClustering.from_json(groups_asjson)

    def _get_key(self, file_name):
        # TODO: add docstring
        # basename without extension (discards path components if needed)
        basename, ext = os.path.splitext(os.path.basename(file_name))
        # get original name without extension
        source_name = basename.rsplit(sep='_', maxsplit=1)[0]
        # returns name with extension
        return source_name + ext

    def _get_output_path(self, **kwargs):
        # TODO: add docstring
        # TODO: precompute output template during init or in static attribute
        # output path template
        terms = ('{output_dir}', '{file_name}')
        output_path_tpl = os.path.join(*terms)
        return output_path_tpl.format(**kwargs)

    def _store(self, iface_pairs, output_path):
        # TODO: add docstring
        self._log.debug('Storing face pairs in %s' % output_path)

        iface_pairs_json = [(f1.to_json(), f2.to_json())
                             for f1, f2 in iface_pairs]

        with open(output_path, 'w') as json_file:
            json.dump(iface_pairs_json, json_file, indent=4)


def main():
    entry_msg = 'Create pair of faces in the same image for social segments.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing the groups information.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store the face pairs.')
    parser.add_argument('--groups_file_name', default='grouped_faces.json',
                        help='File name containing the groups information.')
    parser.add_argument('--pairs_file_name', default='face_pairs.json',
                        help='File name containing the face pairs.')

    args = parser.parse_args()

    setup_logging(config.LOGGING_CONFIG)

    helper = CreateFacePairsHelper(args.input_dir, args.output_dir,
                                   args.groups_file_name, args.pairs_file_name)
    helper.process_all()


if __name__ == '__main__':
    main()