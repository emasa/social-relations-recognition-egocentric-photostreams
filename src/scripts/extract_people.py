#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os

from egosocial.person.detection import NaivePersonDetector
from egosocial.utils.image_processing import crop_image
from egosocial.utils.parser import FACE_DETECTION, load_faces_from_file


def list_detection_files(directory, file_pattern='*.txt'):
    '''
    List detection files for social segments.
    :param directory: base directory containing detection files. Must be in the
    social segments tree structure: directory / segm_id / {file}

    :return: list of detection files
    '''
    segments_tree = os.path.join('*', file_pattern)
    return glob.glob(os.path.join(directory, segments_tree))


def load_faces_from_directory(directory, method=FACE_DETECTION.DOCKER_FACE):
    pattern = FACE_DETECTION.get_file_pattern(method)
    for detection_path in list_detection_files(directory, file_pattern=pattern):
        faces = load_faces_from_file(detection_path, format=method)
        segm_id = os.path.basename(os.path.dirname(detection_path))
        yield faces, segm_id


from scipy.misc import imsave  # imread
from scipy.ndimage import imread


class BodyFaceImageGenerator:

    def __init__(self, face_detection_method=FACE_DETECTION.DOCKER_FACE):
        self._detector = NaivePersonDetector(3, 5)
        self._face_detection_method = face_detection_method
        self._setup_log()

    def process_directory(self, input_dir, output_dir, detection_dir):
        self._log.info('Starting generation of body-face images')
        # sanity check for input_dir, detection_dir
        self._check_required_directory(input_dir, 'Input')
        self._check_required_directory(detection_dir, 'Detection')
        # generate output directory if necessary
        self._create_directory(output_dir, 'Output', warn_if_exists=True)

        for faces, segm_id in load_faces_from_directory(detection_dir,
                                                        self._face_detection_method):
            if not faces:  # skip loop if no faces found
                self._log.warning('No faces found in segment %s' % segm_id)
                continue

            # skip segments, TODO: just for debugging
            if segm_id not in ['84']:
                self._log.warning('Skip segment. %s' % segm_id)
                continue

            # generate body and face output directories for a given segment
            for image_type in ('body', 'face'):
                segm_output_dir = os.path.join(output_dir, image_type, segm_id)
                self._create_directory(segm_output_dir, 'Segment')

            # get original image name
            image_basename = faces[0].params['image_name']
            # TODO: Workaround! Using dockerface generated images.
            # FIXME: images in sRGB format get unwanted rotation/cropping
            #            image_path_in = os.path.join(input_dir, segm_id,
            # 'dockerface-' + image_basename)
            image_path_in = os.path.join(input_dir, segm_id, image_basename)
            self.process_all_faces(image_path_in, output_dir, segm_id, faces)

    def process_all_faces(self, image_path_in, output_dir, segm_id, faces):
        image_basename = os.path.basename(image_path_in)
        self._log.debug('Loading image %s' % image_basename)
        # read whole image
        image = imread(image_path_in)
        # split actual name and extension
        image_name, image_ext = os.path.splitext(image_basename)

        terms = ('{output_dir}', '{body_face_type}', '{s_id}',
                 '{im_name}_{rank_id}{ext}')
        image_path_out_tpl = os.path.join(*terms)

        for rank_id, face in enumerate(faces):
            self._log.debug(
                'Processing face {} / {}'.format(rank_id + 1, len(faces)))
            # extract body and face subimages & pack them to factorize code
            body_face_pair = zip(('body', 'face'),
                                 self._extract_person(image, face))
            # save images to directories output_dir/{body,image}
            for im_type, new_image in body_face_pair:
                kwargs = dict(output_dir=output_dir, body_face_type=im_type,
                              s_id=segm_id, im_name=image_name, rank_id=rank_id,
                              ext=image_ext)
                # generate a new image's name by appending the rank_id
                image_path_out = image_path_out_tpl.format(**kwargs)
                self._log.debug('Saving image %s' % image_path_out)
                imsave(image_path_out, new_image)

    def _extract_person(self, image, face):
        person = self._detector.detect(image, face)
        self._log.debug('Face bbox {}'.format(face.bbox))
        self._log.debug('Person bbox {}'.format(person.bbox))

        return crop_image(image, person.bbox), crop_image(image, face.bbox)

    def _setup_log(self):
        self._log = logging.getLogger(__file__)
        self._log.setLevel(logging.DEBUG)
        # create console handler
        ch = logging.StreamHandler()
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # add the handlers to logger
        self._log.addHandler(ch)

    def _check_required_directory(self, directory, description=''):
        error_msg_fmt = '%s directory does not exist %s'
        if not (os.path.exists(directory) and os.path.isdir(directory)):
            error_msg = error_msg_fmt % (description, directory)
            self._log.error(error_msg)
            raise NotADirectoryError(error_msg)

        self._log.debug('%s directory: %s' % (description, directory))

    def _create_directory(self, directory, description='',
                          warn_if_exists=False):
        # generate directory if necessary
        if not (os.path.exists(directory) and os.path.isdir(directory)):
            self._log.debug(
                'Creating %s directory %s' % (description, directory,))
            os.makedirs(directory)
        elif warn_if_exists:
            self._log.warning(
                '%s directory already exists %s' % (description, directory))


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

    args = parser.parse_args()

    # TODO: sanity check directories
    generator = BodyFaceImageGenerator(args.detection_fmt)

    generator.process_directory(args.input_dir, args.output_dir,
                                args.detection_dir)


if __name__ == '__main__':
    main()
