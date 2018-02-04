# !/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import logging
import os


def list_segments(directory):
    for root, segments, files in os.walk(directory):
        return segments


def list_files_grouped_by_segment(directory, file_pattern='*.txt',
                                  output_segment_id=False):
    '''
    List files in social segments grouped by segment.
    :param directory: base directory containing detection files. Must be in the
    social segments tree structure: directory / segment_id / {file}

    :return: for each segment, yield ([files in segment], segment_id) if
             output_segment_id=True. Otherwise, yield [files in segment].
    '''
    for segm in list_segments(directory):
        segm_path = os.path.join(directory, segm)
        files = glob.glob(os.path.join(segm_path, file_pattern))
        yield (files, segm) if output_segment_id else files


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


def list_files_in_segment(directory, segment_id, file_pattern='*.txt'):
    segment_dir = os.path.join(directory, segment_id)
    check_directory(segment_dir, 'Segment')
    files = glob.glob(os.path.join(segment_dir, file_pattern))

    return files


def check_directory(directory, description=''):
    log = logging.getLogger(os.path.basename(__name__))
    if not (directory and os.path.isdir(directory)):
        error_msg_fmt = '{} directory does not exist {}'
        error_msg = error_msg_fmt.format(description, directory)
        log.error(error_msg)
        raise NotADirectoryError(error_msg)
    else:
        log.debug('Checked {} directory: {}'.format(description, directory))


def create_directory(directory, description='', warn_if_exists=False):
    log = logging.getLogger(os.path.basename(__name__))
    # generate directory if necessary
    if not directory:
        error_msg_fmt = '{} directory is invalid {}'
        error_msg = error_msg_fmt.format(description, directory)
        raise NotADirectoryError(error_msg)

    if not os.path.isdir(directory):
        log.debug('Creating {} directory {}'.format(description, directory))
        os.makedirs(directory)
    elif warn_if_exists:
        log.warning('{} directory already exists {}'.format(description,
                                                            directory))