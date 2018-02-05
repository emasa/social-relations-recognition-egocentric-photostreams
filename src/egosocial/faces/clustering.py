#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import time

import cognitive_face as cf
import retry

from ..core.types import IdentifiedFace, FaceClustering

# MCS API groups up to 1000 faces per request.
N_FACES_API_LIMIT = 1000


class MCSFaceClustering:
    """ Create group of faces by similarity using Microsoft Cognitive Services.
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
        # set up logging
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

    def __call__(self, face_list):
        """ Group list of faces. Uses callable notation.

            Calls to group method.
        Args:
            :param face_list: list of identified faces.
        Returns:
            :return: List of groups of similar faces.
            :return: Faces without a well-defined group.
            :return: Faces without a valid id to be used in MCS API.
        """
        return self.group(face_list)

    def group(self, face_list):
        """ Group list of faces.

        Args:
            :param face_list: list of identified faces.

        Returns:
            :return: List of groups of similar faces.
            :return: Faces without a well-defined group.
            :return: Faces without a valid id to be used in MCS API.
        """
        # TODO: copy list to allow iterators ?
        self._log.debug('{} faces to group'.format(len(face_list)))

        # gather internal ids, skip faces without the id
        internal_ids = [iface.face_id for iface in face_list
                        if iface.face_id is not None]
        self._log.debug('{} faces with API id'.format(len(internal_ids)))
        # create groups of ids
        grouped_ids, messy_ids = self._group_ids(internal_ids)
        # create groups of faces (map face_id to face)
        groups, messy_group, unknown_group = self._create_face_groups(
            face_list, grouped_ids, messy_ids)

        n_images_grouped = len(internal_ids) - len(messy_group)
        self._log.debug('Found {} groups containing {} faces'.format(
            len(groups), n_images_grouped))
        self._log.debug('{} faces in messy group'.format(len(messy_group)))
        self._log.debug('{} faces without id'.format(len(unknown_group)))

        return FaceClustering(groups, messy_group, unknown_group)

    def _group_ids(self, face_id_list):
        """ Group list of face ids using Microsoft Cognitive Service API.

        Args:
            :param face_id_list: List of valid face_id to be used with MCS API.
        Returns:
            :return: List containing groups of face ids from similar faces.
            :return: List of face ids without a well-defined group.
        """
        # FIXME: remove max limit
        assert len(face_id_list) <= N_FACES_API_LIMIT

        # TODO: implement multiple calls to groups
        # process batch of faces, respecting API limits
        face_id_batch = face_id_list
        grouped_ids, messy_ids = self._group_face_id_batch(face_id_batch)

        return grouped_ids, messy_ids

    def _group_face_id_batch(self, face_id_batch):
        # remove batches with less than two faces
        if len(face_id_batch) < 2:
            self._log.warning('Minimum of two faces per batch. Got {}.'
                              .format(len(face_id_batch)))
            return [], face_id_batch

        self._log.debug('Grouping batch of {} faces'.format(len(face_id_batch)))
        # retry on CognitiveFaceException, sleep 15, 30, 60... seconds
        @retry.retry(cf.util.CognitiveFaceException, logger=self._log,
                     tries=3, delay=15, backoff=2, max_delay=60)
        def retry_group_face_id_batch(_self, face_id_batch):
            # wait enough time to avoid problems with MCS API limits
            if _self._force_wait:
                time.sleep(_self._wait_sec)

            result = cf.face.group(face_id_batch)
            return result['groups'], result['messyGroup']

        return retry_group_face_id_batch(self, face_id_batch)

    def _create_face_groups(self, face_list, grouped_ids, messy_ids):
        """ Get identified faces from ids.

        Args:
            :param face_list: List of identified faces.
            :param grouped_ids: groups of ids with similar face.
            :param messy_ids: ids of faces without a well-defined group.
        Returns:
            :return: List of groups of similar faces.
            :return: Faces without a well-defined group.
            :return: Faces without a valid id to be used in MCS API.
        """
        # map internal face_id to face
        face_map = {iface.face_id: iface for iface in face_list
                                             if iface.face_id is not None}

        # FIXME: workaround, IdentifiedFace is inmutable
        def update_iface(iface, group_id):
            return IdentifiedFace(image_name=iface.image_name,
                                  face_id=iface.face_id,
                                  group_id=group_id)

        # detected groups, set group id (follows results order)
        groups = [[update_iface(face_map[face_id], group_id)
                  for face_id in group]
                  for group_id, group in enumerate(grouped_ids)]

        # faces that couldn't be grouped
        messy_group = [face_map[face_id] for face_id in messy_ids]
        # faces without id
        if len(face_map) == face_list:
            unknown_group = []
        else:
            # TODO: define clearly what is valid id
            unknown_group = [iface for iface in face_list
                                       if iface.face_id is None]

        return groups, messy_group, unknown_group
