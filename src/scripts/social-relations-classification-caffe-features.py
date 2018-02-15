# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import io
import logging
import os
import pickle
import sys

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import scipy

import egosocial.config
from egosocial.utils.caffe.misc import load_levelDB_as_array
from egosocial.utils.logging import setup_logging

# constants
DOMAIN, RELATION = 'domain', 'relation'
END_TO_END, ATTRIBUTES = 'end_to_end', 'attributes'


def train_model(conf):
    log = logging.getLogger(os.path.basename(__file__))

    # load features
    features = get_features(conf)
    # list attributes
    all_attributes = sorted(features['test'].keys())

    log.info('Found {} attributes. List: '.format(len(all_attributes)))
    for attribute in all_attributes:
        log.info('{}'.format(attribute))

    attribute_selector = AttributeSelector(all_attributes)

    for attributes_query in get_model_configurations(all_attributes, conf):
        # all / face / body / or single attributes
        log.info('Selected attribute(s): {}'.format(attributes_query))
        # expand all / face / body / single attribute
        selected_attributes = attribute_selector.filter(attributes_query)
        # get data splits composed by selected attributes only
        data_split = get_data_split(features, selected_attributes,
                                    conf.LABEL_FILES)
        x_test, x_train, x_val, y_test, y_train, y_val = data_split
        # normalize data ?
        normalize = False
        if normalize:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            x_val = scaler.transform(x_val)
        # (pre)trained SVM path to load/save from/to disk
        model_file = get_filename(conf, clf='sgd_loss_squared_hinge',
                                  attrs=attributes_query, ext='.b')
        load_model_path = os.path.join(conf.SVM_MODELS_DIR, model_file)

        # automatically computed (don't modify)
        compute_model = not conf.REUSE_MODEL

        # load model from disk when possible
        if conf.REUSE_MODEL:
            if os.path.exists(load_model_path):
                log.info("Loading precomputed model from disk: {}".format(
                    load_model_path))
                with open(load_model_path, 'rb') as f:
                    clf = pickle.load(f)
            else:
                # if model is not found, recompute from scratch
                print("File not found: {}".format(load_model_path))
                compute_model = True

        # train model from scratch
        if compute_model:
            clf = SGDClassifier(loss="squared_hinge", alpha=0.0001,
                                max_iter=conf.MAX_ITERS, n_jobs=-1,
                                average=True, tol=1e-3,
                                class_weight=conf.CLASS_WEIGHT)

            log.info("Training SGD (SVM loss) from scratch...")
            clf.fit(x_train, y_train)

        log.info(str(clf))

        # save trained model in serialized binary format
        if conf.SAVE_MODEL:
            # a different name could be used
            store_model_path = os.path.join(conf.SVM_MODELS_DIR, model_file)

            if not os.path.isdir(conf.SVM_MODELS_DIR):
                os.mkdir(conf.SVM_MODELS_DIR)

            log.info("Saving model to {}...".format(store_model_path))

            with open(store_model_path, 'wb') as f:
                pickle.dump(clf, f)

        # compute model statistics
        val_stats = compute_stats(x_val, y_val, clf)
        test_stats = compute_stats(x_test, y_test, clf)

        # log statistics
        with io.StringIO() as file_like_str:
            print_statistics(val_stats=val_stats, test_stats=test_stats,
                             fdesc=file_like_str)
            log.info(file_like_str.getvalue())

        # store statistics in disk
        if conf.SAVE_STATS:
            # create directory if it doesn't exist already
            if not os.path.isdir(conf.STATS_MODELS_DIR):
                os.mkdir(conf.STATS_MODELS_DIR)
            stats_file = get_filename(conf, clf='sgd_loss_squared_hinge',
                                      attrs=attributes_query, ext='.txt')
            stats_path = os.path.join(conf.STATS_MODELS_DIR, stats_file)

            with open(stats_path, 'wt') as f:
                log.info("Saving statistics in {}...".format(stats_path))
                print_statistics(val_stats=val_stats, test_stats=test_stats,
                                 fdesc=f)

        # compare with a random forest
        train_rf = False
        if train_rf:
            clf_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                            class_weight=conf.CLASS_WEIGHT)
            print("Training Random Forest from scratch...")
            clf_rf.fit(x_train, y_train)

            with io.StringIO() as file_like_str:
                print_statistics(
                    test_stats=compute_stats(x_test, y_test, clf_rf),
                    fdesc=file_like_str
                )

                log.info(file_like_str.getvalue())


def get_filename(conf, **kwargs):
    # defined classifier format
    # prefix: as needed (e.g. query) | mtype: end2end or attr |
    # nnarch: e.g. caffeNet or VGG | dtype: domain or relation |
    # clf: description of the classifier |
    # ext: file extension (b for serialized binary objects, txt for text)
    # HINT: ordering chosen to match FEATURES_DIR format
    file_fmt = '{prefix}_{mtype}_{layer}_{dtype}_{nnarch}_{clf}{ext}'

    # fill file format
    model_type = 'end2end' if conf.IS_END2END else 'attr'
    balance_descr = 'balanced' if conf.CLASS_WEIGHT else 'unbalanced'
    clf_description = '{}_epochs_{}_{}'.format(kwargs['clf'],
                                               conf.MAX_ITERS,
                                               balance_descr)

    filename = file_fmt.format(prefix=kwargs['attrs'],
                                  mtype=model_type,
                                  layer=conf.LAYER,
                                  nnarch=conf.ARCH,
                                  dtype=conf.DATA_TYPE,
                                  clf=clf_description,
                                  ext=kwargs['ext'])

    return filename


class AttributeSelector:

    def __init__(self, all_attrs):

        body_attributes = self.filter_by_keyword(all_attrs, 'body')
        face_attributes = self.filter_by_keyword(all_attrs, 'face')
        face_attributes.extend(self.filter_by_keyword(all_attrs, 'head'))

        self._selector = {'all': all_attrs,
                          'body': body_attributes,
                          'face': face_attributes}

    def filter(self, query):
        if query in self._selector:
            selected_attributes = self._selector[query]
        else:
            selected_attributes = self.filter_by_keyword(self._selector,
                                                         query)

        return selected_attributes

    def filter_by_keyword(self, attribute_list, key):
        return [attr_name for attr_name in attribute_list if key in attr_name]


def get_model_configurations(all_attributes, conf):
    # extend list of attributes
    if conf.IS_END2END:
        extended_attrs = all_attributes
    else:
        extended_attrs = ['all', 'body', 'face'] + all_attributes

    # some attributes are splitted in two files (one for each person)
    # create a list unique attributes name
    unique_attrs = set(
        [attr[:-2] if attr.endswith('_1') or attr.endswith('_2') else attr
         for attr in extended_attrs])

    return [item for item in sorted(unique_attrs)]


class Configuration:
    def __init__(self, args):

        self.DATA_TYPE = args.data_type
        self.ARCH = 'caffeNet'
        self.LAYER = 'fc7'

        self.CONFIG = '{}_{}_{}'.format(self.LAYER, self.DATA_TYPE, self.ARCH)

        # setup directories
        self.PROJECT_DIR = args.project_dir
        self.BASE_MODELS_DIR = os.path.join(self.PROJECT_DIR,
                                            'models/trained_models')
        self.ATTR_MODELS_DIR = os.path.join(self.BASE_MODELS_DIR,
                                            'attribute_models')
        self.SVM_MODELS_DIR = os.path.join(self.PROJECT_DIR,
                                           'models/svm_models')

        self.SPLITS_DIR = os.path.join(self.PROJECT_DIR,
                                       'datasets/splits/annotator_consistency3')

        self.STATS_MODELS_DIR = os.path.join(self.SVM_MODELS_DIR, 'stats')

        if self.DATA_TYPE == DOMAIN:
            LABEL_FILE_FMT = 'domain_single_body1_{}_5.txt'
        else:
            LABEL_FILE_FMT = 'single_body1_{}_16.txt'

        self.LABEL_FILES = {split: os.path.join(self.SPLITS_DIR,
                                                LABEL_FILE_FMT.format(split))
                            for split in ('train', 'test', 'eval')}

        self.IS_END2END = (args.model_type == END_TO_END)

        self.BASE_FEATURES_DIR = os.path.join(self.PROJECT_DIR,
                                              'extracted_features')
        if self.IS_END2END:
            self.FEATURES_DIR = os.path.join(self.BASE_FEATURES_DIR,
                                             'end_to_end_features',
                                             self.CONFIG)
        else:
            self.FEATURES_DIR = os.path.join(self.BASE_FEATURES_DIR,
                                             'attribute_features',
                                             self.CONFIG)
        self.STORED_FEATURES_DIR = os.path.join(self.FEATURES_DIR,
                                                'all_splits_numpy_format')

        self.PROCESS_FEATURES = args.port_features

        # max number of iterations
        self.MAX_ITERS = 1000

        # deal with unbalanced classes by using penalization weights
        if args.fix_class_unbalance:
            self.CLASS_WEIGHT = 'balanced'
        else:
            self.CLASS_WEIGHT = None

        # reuse precomputed model?
        self.REUSE_MODEL = args.reuse_model
        # save model to disk?
        self.SAVE_MODEL = args.save_model
        # save model statistics to disk?
        self.SAVE_STATS = args.save_stats


def get_data_split(attribute_features, selected_attributes, label_files):
    # concatenate attributes
    fused_features, labels = {}, {}
    for split in ['train', 'test', 'eval']:
        selected_features = [attribute_features[split][attr_name]
                             for attr_name in selected_attributes]

        fused_features[split] = np.concatenate(selected_features, axis=1)

        with open(label_files[split]) as file_label_list:
            labels[split] = np.array([file_label.split()[1]
                                      for file_label in file_label_list],
                                     dtype=np.int)
    # define splits
    X_train = fused_features['train']
    y_train = labels['train']
    X_test = fused_features['eval']
    y_test = labels['eval']
    X_val = fused_features['test']
    y_val = labels['test']

    return X_test, X_train, X_val, y_test, y_train, y_val


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


def compute_stats(X, y, clf):
    y_predicted = clf.predict(X)
    acc = sklearn.metrics.accuracy_score(y, y_predicted)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_predicted)
    report = sklearn.metrics.classification_report(y, y_predicted)

    return acc, confusion_matrix, report


def print_statistics(val_stats=None, test_stats=None, fdesc=sys.stdout):
    for description, stats in [('Validation set:', val_stats),
                               ('Test set:', test_stats)]:

        if stats is not None:
            print(description, file=fdesc)
            accuracy, confusion_matrix, report = stats
            print('Confusion matrix:', file=fdesc)
            print(confusion_matrix, file=fdesc)
            print(file=fdesc)
            print(report, file=fdesc)
            print('SGD accuracy: {:.3f}'.format(accuracy), file=fdesc)
            print('------------------------------------------------',
                  file=fdesc)


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


def load_features(features_dir):
    attribute_features = {split: {} for split in ['train', 'test', 'eval']}

    for numpy_file in os.listdir(features_dir):
        # Split the extension from the path and normalise it to lowercase.
        filename, ext = os.path.splitext(numpy_file)
        ext = ext.lower()

        # path
        features_path = os.path.join(features_dir, numpy_file)

        if ext == '.npz':
            with np.load(features_path) as data:
                features = data['arr_0']
        elif ext == '.npy':
            features = np.load(features_path)
        else:
            continue

        attr_name, split = filename.rsplit('_', 1)
        attribute_features[split][attr_name] = features

        print("Loading {}...".format(features_path))

    return attribute_features


def main():
    setup_logging(egosocial.config.LOGGING_CONFIG)

    entry_msg = 'Reproduce experiments in Social Relation Recognition paper.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--project_dir', required=True,
                        help='Base directory.')

    parser.add_argument('--data_type', required=True,
                        choices=[RELATION, DOMAIN],
                        help='Data type.')

    parser.add_argument('--model_type', required=True,
                        choices=[END_TO_END, ATTRIBUTES],
                        help='Model type.')

    parser.add_argument('--port_features', required=False,
                        action='store_true',
                        help='Whether port features from other formats to'
                             'numpy.')

    parser.add_argument('--fix_class_unbalance', required=False,
                        action='store_true',
                        help='Use scikit automatic method to deal with class'
                             'unbalance.')

    parser.add_argument('--reuse_model', required=False,
                        action='store_true',
                        help='Use precomputed model if available.')

    parser.add_argument('--save_model', required=False,
                        action='store_true',
                        help='Save model to disk.')

    parser.add_argument('--save_stats', required=False,
                        action='store_true',
                        help='Save statistics to disk.')

    args = parser.parse_args()
    # keep configuration
    conf = Configuration(args)

    train_model(conf)
