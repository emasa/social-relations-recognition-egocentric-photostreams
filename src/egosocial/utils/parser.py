#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from ..core.types import BBox, Face


class FACE_DETECTION:
    DOCKER_FACE = 'dockerface'
    MCS = 'mcs'
    FILE_PATTERNS = {DOCKER_FACE: '*.txt', MCS: '*.txt'}

    @classmethod
    def check_method(cls, detection_method):
        '''
        Check face detection method. Options: dockerface, mcs.
        Raise NotImplementedError in case of invalid input.
        :param detection_method
        '''
        if not cls.is_valid(detection_method):
            error_msg = 'Invalid face detection method: {}. Valid ' \
                        'options: {}.'
            valid_str = ','.join(cls.get_valid_formats())
            raise NotImplementedError(
                error_msg.format(detection_method, valid_str))

    @classmethod
    def is_valid(cls, detection_method):
        return detection_method in cls.FILE_PATTERNS

    @classmethod
    def get_file_pattern(cls, detection_method):
        cls.check_method(detection_method)
        return cls.FILE_PATTERNS[detection_method]

    @classmethod
    def get_valid_formats(cls):
        return sorted(cls.FILE_PATTERNS.keys())


def load_faces_from_mcs_format(detection_file):
    faces = []

    with open(detection_file) as faces_json:
        for face_info in json.loads(faces_json.read()):
            bbox_dict = face_info['faceRectangle']
            bbox = BBox(**bbox_dict)
            face = Face(bbox=bbox, params=face_info)
            faces.append(face)

    return faces


def load_faces_from_facedocker_format(detection_file):
    faces = []

    with open(detection_file) as faces_text:
        for face_line in faces_text:
            image_name, *coordinates, confidence_score = face_line.strip(

            ).split()

            # rounds float number to the next smaller integer
            x_min, y_min, x_max, y_max = [int(float(c)) for c in coordinates]
            # convert coordinates to bbox format
            top, left, height, width = y_min, x_min, y_max - y_min, x_max - \
                                       x_min
            bbox = BBox(top, left, height, width)
            # keep extra parameters
            params = {'image_name': image_name,
                      'confidence_score': float(confidence_score)}

            face = Face(bbox=bbox, params=params)
            faces.append(face)

    return faces


def load_faces_from_file(detection_file, format='dockerface'):
    if format == 'dockerface':
        return load_faces_from_facedocker_format(detection_file)
    elif format == 'mcs':
        return load_faces_from_mcs_format(detection_file)
    else:
        error_msg = 'Format {} not implemented. Valid formats: dockerface, mcs.'
        raise NotImplementedError(error_msg % format)
