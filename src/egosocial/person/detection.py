#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from ..core.types import BBox, Person


class NaivePersonDetector:
    def __init__(self, width_multiplier=3, height_multiplier=6):
        self.width_multiplier = width_multiplier
        self.height_multiplier = height_multiplier

        # TODO: raise exception
        assert self.width_multiplier > 1 and self.height_multiplier > 1

    def detect(self, image, face):
        bbox = self._compute_bbox(image, face)
        person = Person(bbox=bbox, face=copy.deepcopy(face), params=None)

        return person

    def detect_all(self, image, faces):
        return [self.detect(image, face) for face in faces]

    def _compute_bbox(self, image, face):
        # image size
        H, W = image.shape[0:2]
        f_top, f_left, f_height, f_width = face.bbox
        # TODO: Review implementation
        # person coordinates
        height = int(f_height * self.height_multiplier)
        width = int(f_width * self.width_multiplier)

        height_offset = int((height - face.bbox.height) / 10.0)
        top = f_top - height_offset

        width_offset = int((width - face.bbox.width) / 2.0)
        left = f_left - width_offset

        # checking the boundaries
        top, left = min(max(top, 0), H), min(max(left, 0), W)
        height, width = min(height, H - top), min(width, W - left)

        return BBox(top, left, height, width)
