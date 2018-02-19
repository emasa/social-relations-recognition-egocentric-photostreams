# !/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import scipy

from ..utils.caffe.misc import load_levelDB_as_array


def load_features(features_dir, filename_cb, log=None):
    if log is None:
        log = logging.getLogger(os.path.basename(__file__))

    # splits (switch from caffe's split name convention to keras's convention)
    _train, _val, _test = 'train', 'test', 'eval'

    attribute_features = {split: {} for split in (_train, _val, _test)}

    for numpy_file in sorted(os.listdir(features_dir)):
        split, attr_name, ext = filename_cb(numpy_file)

        # absolute path
        features_path = os.path.join(features_dir, numpy_file)

        if ext in ('.npz', '.npy'):
            log.debug("Loading {}...".format(features_path))
        else:
            log.warning("Found file with unknown format.".format(features_path))
            continue # skip this file

        if ext == '.npz':
            with np.load(features_path) as data:
                features = data['arr_0']
        else:
            features = np.load(features_path)

        # fuse attributes in several files
        if attr_name in attribute_features[split]:
            array = np.concatenate([attribute_features[split][attr_name],
                                    features.astype('float32', copy=False)],
                                   axis=1)
        else:
            array = features.astype('float32', copy=False)

        attribute_features[split][attr_name] = array

    return attribute_features

def preprocess_attributes(levelDB_dirs=None, raw_numpy_dirs=None,
                          matlab_dirs=None):
    splits = ['train', 'test', 'eval']
    attribute_features = {split: {} for split in splits}

    ###########################################################################
    # features in levelDB format
    if levelDB_dirs:
        for directory in levelDB_dirs:
            for split in splits:
                attribute_models = os.listdir(os.path.join(directory, split))

                for attr_name in attribute_models:
                    features = load_levelDB_as_array(
                        os.path.join(directory, split, attr_name))
                    attribute_features[split][attr_name] = features
                    msg = "Convert from levelDB dataset: {} attribute: {} " \
                          "dim: {}"
                    print(msg.format(split, attr_name, features.shape))

    ###########################################################################
    # features in numpy format
    if raw_numpy_dirs:
        for directory in raw_numpy_dirs:
            for numpy_file in os.listdir(directory):
                filename, ext = os.path.splitext(numpy_file)
                if ext.lower() == '.npy':
                    # load numpy
                    features = np.load(os.path.join(directory, numpy_file))
                    # find split
                    for candidate_split in splits:
                        if candidate_split in filename:
                            split = candidate_split
                            break
                    else:
                        split = None
                    # the folder name is the attribute name
                    attr_name = os.path.basename(directory)

                    attribute_features[split][attr_name] = features
                    msg = "Load numpy format dataset: {} attribute: {} dim: {}"
                    print(msg.format(split, attr_name, features.shape))

    ###########################################################################
    # features in matlab format
    if matlab_dirs:
        for directory in matlab_dirs:
            for matfile in os.listdir(directory):
                filename, ext = os.path.splitext(matfile)
                if ext.lower() == '.mat':
                    # load matfile (dict format)
                    matfile_dict = scipy.io.loadmat(
                        os.path.join(directory, matfile))
                    attr_name, split = filename.rsplit('_', 1)
                    # access numpy field
                    features = matfile_dict[attr_name]
                    attribute_features[split][attr_name] = features
                    msg = "Convert matlab format dataset: {} attribute: {} " \
                          "dim: {}"
                    print(msg.format(split, attr_name, features.shape))

    ###########################################################################

    return attribute_features


def save_features(attribute_features, features_dir, compressed=True):
    if not (os.path.exists(features_dir) and os.path.isdir(features_dir)):
        os.mkdir(features_dir)

    for split, attributes in attribute_features.items():
        for attr_name, features in attributes.items():
            features_path = os.path.join(features_dir, '{}_{}').format(
                attr_name, split)

            # save file in compress format and float16
            if compressed:
                np.savez_compressed(features_path, features.astype(np.float16))
            else:
                np.save(features_path, features)

            print("Saved {}.{} ...".format(features_path,
                                           'npz' if compressed else 'np'))


def get_features(conf):
    # preprocess features from original formats (leveldb, numpy, matlab)
    if conf.PROCESS_FEATURES:
        if conf.IS_END2END:
            # LEVELDB DIRS
            levelDB_dirs = [conf.FEATURES_DIR]
            attribute_features = preprocess_attributes(levelDB_dirs)
        else:
            # LEVELDB DIRS
            levelDB_dirs = [conf.FEATURES_DIR]
            # MATLAB DIRS
            matlab_dirs = [
                os.path.join(conf.ATTR_MODELS_DIR,
                             'localation_scale_data(annotator_consistency3)')
            ]
            # NUMPY DIRS
            numpy_dirs = [
                os.path.join(conf.ATTR_MODELS_DIR,
                             'imsitu_body_activity(annotator_consistency3)'),
                os.path.join(conf.ATTR_MODELS_DIR,
                             'body_immediacy(annotator_consistency3)')
            ]

            attribute_features = preprocess_attributes(levelDB_dirs, numpy_dirs,
                                                       matlab_dirs)

        if not (os.path.isdir(conf.STORED_FEATURES_DIR)):
            os.mkdir(conf.STORED_FEATURES_DIR)

        # save features to disk
        save_features(attribute_features, conf.STORED_FEATURES_DIR,
                      compressed=True)

    else:
        # load features from disk
        attribute_features = load_features(conf.STORED_FEATURES_DIR)

    return attribute_features