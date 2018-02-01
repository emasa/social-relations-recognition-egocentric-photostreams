#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pprint

import cognitive_face as CF

from src.egosocial.faces import detection

if __name__ == '__main__':
    with open('./credentials.json.nogit') as credentials_file:
        credentials = json.load(credentials_file)

    KEY = credentials['azure_face_api_key']
    CF.Key.set(KEY)

    face_detector = detection.MCSFaceDetector()

    img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face' \
              '-Windows/master/Data/detection1.jpg'
    faces = face_detector.detect(img_url)

    print('Found {} faces'.format(len(faces)))
    pprint.pprint(faces[0].params)
