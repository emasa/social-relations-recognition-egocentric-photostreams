#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cognitive_face as CF

from ..core.types import BBox, Face
from ..utils.concurrency import RatedSemaphore

DEFAULT_BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1' \
                   '.0/'
CF.BaseUrl.set(DEFAULT_BASE_URL)

FACIAL_ATTRIBUTES = ['age', 'gender', 'headPose', 'smile', 'facialHair',
                     'glasses', 'emotion', 'hair', 'makeup', 'occlusion',
                     'accessories', 'blur', 'exposure', 'noise']


class MCSFaceDetector:
    '''
    Interface to Microsoft Cognitive Services for face detection.
    '''

    def __init__(self, free_tier=True):
        '''

        :param free_tier:
        '''
        self._free_tier = free_tier
        self._landmarks = True
        self._extra_attrs = ','.join(FACIAL_ATTRIBUTES)

        # API limits: max number of requests according to API level
        # Warning: this implemention enforces limits per object instance
        # Using multiple instances in parallel could lead to
        # quota errors (CognitiveFaceException)
        if self._free_tier:
            # free tier: up to 20 requests per minute
            # let's do 18 to avoid problems
            self._rate_limit = RatedSemaphore(value=18, period=60)
        else:
            # up to 10 requests per second
            self._rate_limit = RatedSemaphore(value=10, period=1)

    def _detect(self, image):
        '''
        :param image:
        :return:
        '''
        response = CF.face.detect(image, landmarks=self._landmarks,
                                  attributes=self._extra_attrs)
        faces = []

        for face_info in response:
            bbox_dict = face_info['faceRectangle']
            bbox = BBox(**bbox_dict)
            face = Face(bbox=bbox, params=face_info)
            faces.append(face)

        return faces

    def detect(self, image):
        # API limits
        with self._rate_limit:
            return self._detect(image)

    def detect_all(self, image_list):
        '''

        :param image_list:
        :return:
        '''
        # API limits
        return [self.detect(image) for image in image_list]
