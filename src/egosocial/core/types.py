#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

BBox = collections.namedtuple('BBox', ('top', 'left', 'height', 'width'))


def bbox_from_json(bbox_asjson):
    if isinstance(bbox_asjson, dict):
        bbox_dict = bbox_asjson
    else:
        bbox_dict = dict(zip(('top', 'left', 'height', 'width'), bbox_asjson))

    return BBox(**bbox_dict)


BBox.from_json = bbox_from_json

Face = collections.namedtuple('Face', ('bbox',
                                       'image_name',
                                       'face_id',
                                       'params'))


def face_from_json(face_asjson):
    if isinstance(face_asjson, dict):
        face_dict = dict(face_asjson)
    else:
        face_dict = dict(zip(('bbox', 'image_name', 'face_id', 'params'), face_asjson))

    face_dict['bbox'] = BBox.from_json(face_dict['bbox'])

    return IdentifiedFace(**face_dict)


Person = collections.namedtuple('Person', ('bbox', 'face', 'params'))

IdentifiedFace = collections.namedtuple('IdentifiedFace', ('bbox',
                                                           'image_name',
                                                           'segment_id',
                                                           'face_id',
                                                           'group_id'))

def create_identified_face(bbox=None, image_name=None, segment_id=None,
                           face_id=None, group_id=None):

    return IdentifiedFace(bbox, image_name, segment_id, face_id, group_id)

def identified_face_from_json(iface_asjson):
    if isinstance(iface_asjson, dict):
        iface_dict = dict(iface_asjson)
    else:
        iface_dict = dict(zip(('bbox', 'image_name', 'face_id', 'segment_id',
                               'group_id'), iface_asjson))

    iface_dict['bbox'] = BBox.from_json(iface_dict['bbox'])

    return IdentifiedFace(**iface_dict)


IdentifiedFace.from_json = identified_face_from_json
IdentifiedFace.create = create_identified_face

FaceClustering = collections.namedtuple('FaceClustering', ('groups',
                                                           'messyGroup',
                                                           'unknownGroup'))

def create_face_clustering(groups=None, messyGroup=None, unknownGroup=None):
    return FaceClustering(groups, messyGroup, unknownGroup)

def face_clustering_from_json(groups_asjson):

    if isinstance(groups_asjson, dict):
        groups_dict = groups_asjson
    else:
        groups_dict = dict(groups=groups_asjson[0],
                           messyGroup=groups_asjson[1],
                           unknownGroup=groups_asjson[2])

    return FaceClustering(
              groups=[[IdentifiedFace.from_json(iface) for iface in group]
                      for group in groups_dict['groups']],
              messyGroup=[IdentifiedFace.from_json(iface)
                          for iface in groups_dict['messyGroup']],
              unknownGroup=[IdentifiedFace.from_json(iface)
                            for iface in groups_dict['unknownGroup']]
           )

FaceClustering.create = create_face_clustering
FaceClustering.from_json = face_clustering_from_json