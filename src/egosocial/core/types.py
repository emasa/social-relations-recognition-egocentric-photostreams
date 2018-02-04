#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections

BBox = collections.namedtuple('BBox', ('top', 'left', 'height', 'width'))


def create_bbox(top=None, left=None, height=None, width=None):
    return BBox(top, left, height, width)


def bbox_from_json(bbox_asjson):
    if isinstance(bbox_asjson, dict):
        bbox_dict = bbox_asjson
    else:
        bbox_dict = dict(zip(('top', 'left', 'height', 'width'), bbox_asjson))

    return BBox(**bbox_dict)


def bbox_to_json(bbox):
    return bbox._asdict()


BBox.create = create_bbox
BBox.from_json = bbox_from_json
BBox.to_json = bbox_to_json


Face = collections.namedtuple('Face', ('bbox',
                                       'image_name',
                                       'face_id',
                                       'params'))


def create_face(bbox=None, image_name=None, face_id=None, params=None):
    return Face(bbox, image_name, face_id, params)


def face_from_json(face_asjson):
    if isinstance(face_asjson, dict):
        face_dict = dict(face_asjson)
    else:
        face_dict = dict(zip(('bbox', 'image_name', 'face_id', 'params'), face_asjson))

    if face_dict['bbox']:
        face_dict['bbox'] = BBox.from_json(face_dict['bbox'])

    return Face(**face_dict)


def face_to_json(face):
    face_dict = face._asdict()

    if face_dict['bbox']:
        face_dict['bbox'] = BBox.to_json(face_dict['bbox'])

    return face_dict


Face.create = create_face
Face.from_json = face_from_json
Face.to_json = face_to_json

Person = collections.namedtuple('Person', ('bbox', 'face', 'params'))

IdentifiedFace = collections.namedtuple('IdentifiedFace', ('image_name',
                                                           'face_id',
                                                           'group_id'))


def create_identified_face(image_name=None, face_id=None, group_id=None):

    return IdentifiedFace(image_name, face_id, group_id)


def identified_face_from_json(iface_asjson):
    if isinstance(iface_asjson, dict):
        iface_dict = iface_asjson
    else:
        iface_dict = dict(zip(('image_name', 'face_id', 'group_id'),
                              iface_asjson))

    return IdentifiedFace(**iface_dict)


def identified_face_to_json(iface):
    return iface._asdict()


IdentifiedFace.create = create_identified_face
IdentifiedFace.from_json = identified_face_from_json
IdentifiedFace.to_json = identified_face_to_json

# FIXME: change attribute's name to underscores for consistency
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


def face_clustering_to_json(face_clustering):
    return dict(
            groups=[[IdentifiedFace.to_json(iface) for iface in group]
                    for group in face_clustering.groups],
            messyGroup=[IdentifiedFace.to_json(iface)
                        for iface in face_clustering.messyGroup],
            unknownGroup=[IdentifiedFace.to_json(iface)
                          for iface in face_clustering.unknownGroup]
            )


FaceClustering.create = create_face_clustering
FaceClustering.from_json = face_clustering_from_json
FaceClustering.to_json = face_clustering_to_json