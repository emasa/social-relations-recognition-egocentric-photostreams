#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os

import cv2

from egosocial import config
from egosocial.person.detection import NaivePersonDetector
from egosocial.utils.filesystem import check_directory, create_directory
from egosocial.utils.filesystem import list_files
from egosocial.utils.image_processing import crop_image
from egosocial.utils.logging import setup_logging
from egosocial.utils.parser import FACE_DETECTION, load_faces_from_file


class BodyFaceExtractor:

    def __init__(self, face_detection_method=FACE_DETECTION.DOCKER_FACE):
        self._detector = NaivePersonDetector(3, 5)
        self._face_detection_method = face_detection_method
        # set up logging
        self._log = logging.getLogger(os.path.basename(__file__))
        # default without folders inside segment
        self.segment_input_extra = []

    def process_directory(self, input_dir, output_dir, detection_dir):
        self._log.info('Starting generation of body-face images')
        # sanity check for input_dir, detection_dir
        check_directory(input_dir, 'Input')
        check_directory(detection_dir, 'Detection')
        # generate output directory if necessary
        create_directory(output_dir, 'Output', warn_if_exists=True)

        for faces, segm_id in self._load_faces_from_directory(detection_dir):
            # get original image name
            image_basename = faces[0].image_name

            # deal with inner folders inside segments
            image_path_in_unrolled = [input_dir, segm_id]
            image_path_in_unrolled.extend(self.segment_input_extra)
            image_path_in_unrolled.append(image_basename)

            image_path_in = os.path.join(*image_path_in_unrolled)
            image = self._load_input(image_path_in)
            body_list, face_list = self._process_batch(image, faces)

            # generate body and face output directories for a given segment
            for im_type, images in (('body', body_list), ('face', face_list)):
                segm_output_dir = os.path.join(output_dir, im_type, segm_id)
                create_directory(segm_output_dir, 'Segment')
                self._store(images, segm_output_dir, image_basename)

    def _load_faces_from_directory(self, directory):
        pattern = FACE_DETECTION.get_file_pattern(self._face_detection_method)

        self._log.debug('Faces in %s' % directory)
        for detection_path, segm_id in list_files(directory,
                                                  file_pattern=pattern,
                                                  output_segment_id=True):
            self._log.debug('Loading faces from %s' % detection_path)
            faces = load_faces_from_file(detection_path,
                                         format=self._face_detection_method)
            # skip loop if no faces found
            if not faces:
                self._log.warning('No faces found in file %s segment %s' %
                                  ((detection_path, segm_id)))
                continue

            filtered_faces = self._filter(faces, segm_id, use_confidence=True)

            if not filtered_faces:
                self._log.warning(
                    'No faces after filtering in file %s segment %s' %
                    ((detection_path, segm_id)))
                continue

            yield filtered_faces, segm_id

    def _load_input(self, image_path):
        self._log.debug('Loading image file %s' % image_path)
        # read whole image
        return cv2.imread(image_path)

    def _filter(self, faces, segment_id,
                use_segment_id=False, use_confidence=False):
        filtered_faces = faces

        if use_segment_id:
            allowed_segments = []

            if segment_id not in allowed_segments:
                filtered_faces = []

        if use_confidence:
            min_confidence = 0.99
            filtered_faces = [face for face in filtered_faces
                              if 'confidence' not in face.params or
                              face.params['confidence'] >= min_confidence]

        skipped_faces = len(faces) - len(filtered_faces)
        if skipped_faces:
            self._log.warning('Skip {} {} in segment {}'
                              .format(skipped_faces,
                                      'faces' if skipped_faces > 1 else 'face',
                                      segment_id)
                              )

        return filtered_faces

    def _process_batch(self, image, faces):
        return zip(*[self._extract_person(image, face) for face in faces])

    def _store(self, image_list, output_dir, image_basename):
        image_name, image_ext = os.path.splitext(image_basename)
        for rank_id, output_image in enumerate(image_list):
            # generate a new image's name by appending the rank_id
            output_path = self._get_output_path(output_dir=output_dir,
                                                im_name=image_name,
                                                rank_id=rank_id,
                                                ext=image_ext)
            # save images to output_dir
            self._log.debug('Saving image %s' % output_path)
            cv2.imwrite(output_path, output_image)

    def _extract_person(self, image, face):
        self._log.debug('Processing face {}'.format(face.face_id))

        person = self._detector.detect(image, face)
        self._log.debug('Face bbox {}'.format(face.bbox))
        self._log.debug('Person bbox {}'.format(person.bbox))

        return crop_image(image, person.bbox), crop_image(image, face.bbox)

    def _get_output_path(self, **kwargs):
        # output path template
        terms = ('{output_dir}', '{im_name}_{rank_id}{ext}')
        output_path_tpl = os.path.join(*terms)
        return output_path_tpl.format(**kwargs)


def main():
    entry_msg = 'Extract body and face imaging from people in an image.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--input_dir', required=True,
                        help='Directory containing social segments.')
    parser.add_argument('--detection_dir', required=True,
                        help='Directory containing face detection for social '
                             'segments.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where to store body and face images.')

    parser.add_argument('--detection_fmt',
                        choices=FACE_DETECTION.get_valid_formats(),
                        default=FACE_DETECTION.DOCKER_FACE,
                        help="""Face detection method. Default: dockerface.""")

    parser.add_argument('--input_dir_extra', default='',
                        help="""Extra folders inside input segment directories.
                                Example: given a heirarchy 10/bar/foo/, 
                                call with --input_dir_extra bar,foo
                            """)

    args = parser.parse_args()

    setup_logging(config.LOGGING_CONFIG)

    generator = BodyFaceExtractor(args.detection_fmt)

    extra = args.input_dir_extra
    # work around for test folders, run with '--input_dir_extra data
    # configure extra folders inside input segments (default empty)
    generator.segment_input_extra = extra.strip().split(',') if extra else []

    generator.process_directory(args.input_dir, args.output_dir,
                                args.detection_dir)


if __name__ == '__main__':
    main()
