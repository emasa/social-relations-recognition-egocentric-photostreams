#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

BBox = collections.namedtuple('BBox', ('top', 'left', 'height', 'width'))
Face = collections.namedtuple('Face', ('bbox', 'params'))
Person = collections.namedtuple('Person', ('bbox', 'face', 'params'))
