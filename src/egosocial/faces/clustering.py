#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import cognitive_face as cf

from ..utils.concurrency import RatedSemaphore

# MCS API groups up to 1000 faces per request.
N_FACES_API_LIMIT = 1000


class MCSFaceClustering:
    """ Create group of faces by similarity using Microsoft Cognitive Services.
    """

    def __init__(self, free_tier=True):
        """
        Args:
            :param free_tier (bool): indicates if the MCS free tier is used.
            Free tiers is restricted to up 18 requests per minute (hard limit
            is 20 reqs per min). Paid tier allows up to 10 reqs per second.
        """
        self._free_tier = free_tier
        # set up logging
        self._log = logging.getLogger(os.path.basename(__file__))

        # API limits: max number of requests according to API level
        # Warning: this implementation enforces limits per object instance
        # Using multiple instances in parallel could lead to
        # quota errors (CognitiveFaceException)
        self._log.debug('Using Microsoft Cognitive Services with {} tier'
                        .format('free' if free_tier else 'paid'))
        if self._free_tier:
            # free tier: up to 20 requests per minute
            # let's do 18 to avoid problems
            self._rate_limit = RatedSemaphore(value=18, period=60)
        else:
            # paid access allows up to 10 requests per second
            self._rate_limit = RatedSemaphore(value=10, period=1)

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

        return groups, messy_group, unknown_group

    def _group_ids(self, face_id_list):
        """ Group list of face ids using Microsoft Cognitive Service API.

        Args:
            :param face_id_list: List of valid face_id to be used with MCS API.
        Returns:
            :return: List containing groups of face ids from similar faces.
            :return: List of face ids without a well-defined group.
        """
        # TODO: implement multiple calls to groups
        # FIXME: remove max limit
        assert len(face_id_list) <= N_FACES_API_LIMIT
        # remove empty groups if there aren't faces
        if not face_id_list:
            return [], []
        # process batch of faces, respecting API limits
        batch = face_id_list
        # API limits
        with self._rate_limit:
            self._log.debug('Grouping batch of {} faces'.format(len(batch)))
            # TODO: improve error handling
            result = cf.face.group(batch)

        grouped_ids, messy_ids = result['groups'], result['messyGroup']

        return grouped_ids, messy_ids

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
        # detected groups
        groups = [
            [face_map[face_id] for face_id in group] for group in grouped_ids
        ]
        # faces that couldn't be grouped
        messy_group = [face_map[face_id] for face_id in messy_ids]
        # faces without id
        if len(face_map) == face_list:
            unknown_group = []
        else:
            # TODO: define clearly what is valid id
            unknown_group = [iface for iface in face_list if
                             iface.face_id is None]

        return groups, messy_group, unknown_group
