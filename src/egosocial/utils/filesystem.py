# !/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import logging
import os


def list_files_grouped_by_segment(directory, file_pattern='*.txt',
                                  output_segment_id=False):
    '''
    List files in social segments grouped by segment.
    :param directory: base directory containing detection files. Must be in the
    social segments tree structure: directory / segment_id / {file}

    :return: for each segment, yield ([files in segment], segment_id) if
             output_segment_id=True. Otherwise, yield [files in segment].
    '''
    for root, segments, files in os.walk(directory):
        for segm in segments:
            segm_path = os.path.join(root, segm)
            files = glob.glob(os.path.join(segm_path, file_pattern))
            yield (files, segm) if output_segment_id else files

        # leave the loop
        break


def list_files(directory, file_pattern='*.txt', output_segment_id=False):
    '''
    List files in social segments.
    :param directory: base directory containing segments. Must be in the
    social segments tree structure: directory / segm_id / {file}

    :return: for each segment, for yield ([files in segment], segment_id) if
             output_segment_id=True. Otherwise, yield [files in segment].
    '''
    for files, segm in list_files_grouped_by_segment(directory, file_pattern,
                                                     output_segment_id=True):
        for file in files:
            yield (file, segm) if output_segment_id else file


def check_directory(directory, description=''):
    log = logging.getLogger(os.path.basename(__name__))
    if not (os.path.exists(directory) and os.path.isdir(directory)):
        error_msg_fmt = '%s directory does not exist %s'
        error_msg = error_msg_fmt % (description, directory)
        log.error(error_msg)
        raise NotADirectoryError(error_msg)
        log.debug('%s directory: %s' % (description, directory))
    else:
        log.debug('Checked %s directory: %s' % (description, directory))


def create_directory(directory, description='', warn_if_exists=False):
    log = logging.getLogger(os.path.basename(__name__))
    # generate directory if necessary
    if not (os.path.exists(directory) and os.path.isdir(directory)):
        log.debug('Creating %s directory %s' % (description, directory))
        os.makedirs(directory)
    elif warn_if_exists:
        log.warning('%s directory already exists %s' % (description, directory))
