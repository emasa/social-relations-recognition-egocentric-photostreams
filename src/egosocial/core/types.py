#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

BBox = collections.namedtuple('BBox', ('top', 'left', 'height', 'width'))

Face = collections.namedtuple('Face', ('bbox',
                                       'image_name',
                                       'face_id',
                                       'params'))

Person = collections.namedtuple('Person', ('bbox', 'face', 'params'))

IdentifiedFace = collections.namedtuple('IdentifiedFace', ('bbox',
                                                           'image_name',
                                                           'face_id',
                                                           'segment_id',
                                                           'group_id'))
