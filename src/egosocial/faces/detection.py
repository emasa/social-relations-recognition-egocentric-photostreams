#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import time

import cognitive_face as cf
import retry

from ..core.types import BBox, Face

FACIAL_ATTRIBUTES = ['age', 'gender', 'headPose', 'smile', 'facialHair',
                     'glasses', 'emotion', 'hair', 'makeup', 'occlusion',
                     'accessories', 'blur', 'exposure', 'noise']


class MCSFaceDetector:
    """
    Interface to Microsoft Cognitive Services for face detection.
    """
    def __init__(self, free_tier=True, force_wait=True):
        """
        Args:
            :param free_tier (bool): indicates if the MCS free tier is used.
            Free tier is restricted to 20 requests per minutes. Paid tier
            allows up to 10 reqs per second.
        """
        self._free_tier = free_tier
        self._force_wait = force_wait

        self._landmarks = True
        self._extra_attrs = ','.join(FACIAL_ATTRIBUTES)
        self._log = logging.getLogger(os.path.basename(__file__))

        # API limits: max number of requests according to API level
        # Warning: this implementation enforces limits per object instance
        # Using multiple instances in parallel could lead to
        # quota errors (CognitiveFaceException)
        self._log.debug('Using Microsoft Cognitive Services with {} tier'
                        .format('free' if free_tier else 'paid'))

        if self._free_tier:
            # wait 3 seconds
            self._wait_sec = 3
        else:
            # wait 0.1 second
            self._wait_sec = 0.1

    def _detect(self, image_path):
        """
        :param image:
        :return:
        """
        detection = cf.face.detect(image_path, landmarks=self._landmarks,
                                   attributes=self._extra_attrs)

        faces = [Face(bbox=BBox.from_json(face_dict['faceRectangle']),
                      image_name=os.path.basename(image_path),
                      face_id=face_dict['faceId'], params=face_dict)
                 for face_dict in detection]

        return faces

    def __call__(self, image):
        return self.detect(image)

    def detect(self, image):
        # retry on CognitiveFaceException, sleep 15, 30, 60... seconds
        @retry.retry(cf.util.CognitiveFaceException, logger=self._log,
                     tries=3, delay=15, backoff=2, max_delay=60)
        def retry_detect(_self, image):
            # wait enough time to avoid problems with MCS API limits
            if _self._force_wait:
                time.sleep(_self._wait_sec)

            return _self._detect(image)

        return retry_detect(self, image)

    def detect_all(self, image_list):
        """

        :param image_list:
        :return:
        """
        return [self.detect(image) for image in image_list]
