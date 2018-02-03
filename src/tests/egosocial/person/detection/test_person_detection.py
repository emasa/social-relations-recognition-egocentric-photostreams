#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from scipy.ndimage import imread

from egosocial.config import ASSETS_DIR
from egosocial.person.detection import NaivePersonDetector
from egosocial.utils.parser import load_faces_from_mcs_format


class TestNaivePersonDetector:
    def setup_class(self):
        self.detector = NaivePersonDetector(3, 6)
        self.image = imread(os.path.join(ASSETS_DIR, 'american_family.jpg'))
        self.faces_path = os.path.join(ASSETS_DIR,
                                       'american_family_face_detection_mcs.json')

        self.faces = load_faces_from_mcs_format(self.faces_path)

    def test_detect_base(self):
        face = self.faces[0]
        person = self.detector.detect(self.image, face)

        assert person.bbox.width == 3 * face.bbox.width
        assert person.bbox.height == 6 * face.bbox.height
        assert person.face == self.faces[0]

    def test_detect_all_base(self):
        image_face_list = zip([self.image] * len(self.faces), self.faces)
        people = self.detector.detect_all(image_face_list)

        assert [p.face for p in people] == self.faces

    def test_detect_all_empty(self):
        assert len(self.detector.detect_all([])) == 0

    def test_detect_all_one(self):
        people = self.detector.detect_all(self.image, self.faces[0])

        assert len(people) == 1 and people[0].face == self.faces[0]
