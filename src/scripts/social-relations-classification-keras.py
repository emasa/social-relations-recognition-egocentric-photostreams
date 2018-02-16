# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

import numpy as np
import scipy
import sklearn
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import egosocial.config
from egosocial.core.types import relation_to_domain_vec
from egosocial.utils.caffe.misc import load_levelDB_as_array
from egosocial.utils.keras.autolosses import AutoMultiLossWrapper
from egosocial.utils.logging import setup_logging

# sys.path.extend([os.path.dirname(os.path.dirname(__file__))])

# constants
DOMAIN, RELATION = 'domain', 'relation'
END_TO_END, ATTRIBUTES = 'end_to_end', 'attributes'

N_CLS_RELATION, N_CLS_DOMAIN = 16, 5


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
    attributes_query = 'all'
    # expand all / face / body / single attribute
    selected_attributes = attribute_selector.filter(attributes_query)
    # all / face / body / or single attributes
    log.info('Selected attribute(s): {}'.format(attributes_query))

    # get data splits composed by selected attributes only
    data_split = get_data_split(features, selected_attributes,
                                conf.LABEL_FILES)
    x_train, x_val, x_test, *labels = data_split
    # preprocess the data
    x_train, x_val, x_test = preprocess_data(x_train, x_val, x_test)
    # one-hot encoding for relation
    y_train_rel, y_val_rel, y_test_rel = [
        to_categorical(y, N_CLS_RELATION) for y in labels
    ]
    # one-hot encoding for domain
    y_train_dom, y_val_dom, y_test_dom = [
        to_categorical(relation_to_domain_vec(y), N_CLS_DOMAIN) for y in labels
    ]
    # discard number of samples
    input_shape = x_train.shape[1:]
    clf_wrapper = AutoMultiLossWrapper(get_model(input_shape))
    # TODO: move away
    auto_loss = lambda y_true, y_pred: -K.log(categorical_crossentropy(y_true,
                                                                       y_pred))

    clf_wrapper.compile(optimizer='adam', loss=auto_loss, loss_weights='auto')
    log.info(clf_wrapper.model.summary())

    x_train_inputs = {'attribute_features': x_train}
    y_train_outputs = {'relation': y_train_rel, 'domain': y_train_dom}
    x_val_inputs = {'attribute_features': x_val}
    y_val_outputs = {'relation': y_val_rel, 'domain': y_val_dom}
    x_test_inputs = {'attribute_features': x_test}
    y_test_outputs = {'relation': y_test_rel, 'domain': y_test_dom}

    batch_size = conf.BATCH_SIZE
    epochs = conf.EPOCHS

    checkpoint_path = os.path.join(egosocial.config.MODELS_CACHE_DIR,
                                   'multi_attribute',
                                   'weights.{epoch:02d}-{val_loss:.2f}.h5')

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_loss',
                        save_best_only=True),
        TensorBoard(log_dir='./logs', write_images=True, write_graph=True)
    ]

    log.info("Training model from scratch...")
    hist = clf_wrapper.model.fit(
        x_train_inputs, y_train_outputs,
        batch_size=batch_size, epochs=epochs,
        validation_data=(x_val_inputs, y_val_outputs),
        callbacks=callbacks,
    )

    scores = clf_wrapper.model.evaluate(x_test_inputs, y_test_outputs,
                                        batch_size=batch_size)

    log.info(scores)


def get_model(input_shape):
    input_features = Input(shape=input_shape, dtype='float',
                           name='attribute_features')

    relation = Dense(N_CLS_RELATION, activation='linear', name='relation')(
        input_features)
    domain = Dense(N_CLS_DOMAIN, activation='softmax', name='domain')(relation)

    clf = Model(input=input_features, outputs=[domain, relation])

    return clf


def preprocess_data(x_train, x_val, x_test):
    # normalize data ?
    normalize = False
    if normalize:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)

    return x_train, x_val, x_test


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


def get_model_configurations(all_attributes):
    # extend list of attributes
    extended_attrs = ['all', 'body', 'face'] + all_attributes

    # some attributes are splitted in two files (one for each person)
    # create a list unique attributes name
    unique_attrs = set(
        [attr[:-2] if attr.endswith('_1') or attr.endswith('_2') else attr
         for attr in extended_attrs])

    return [item for item in sorted(unique_attrs)]


class Configuration:
    def __init__(self, args):
        self.DATA_TYPE = RELATION
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

        LABEL_FILE_FMT = 'single_body1_{}_16.txt'
        self.LABEL_FILES = {split: os.path.join(self.SPLITS_DIR,
                                                LABEL_FILE_FMT.format(split))
                            for split in ('train', 'test', 'eval')}

        self.IS_END2END = False

        self.BASE_FEATURES_DIR = os.path.join(self.PROJECT_DIR,
                                              'extracted_features')
        self.FEATURES_DIR = os.path.join(self.BASE_FEATURES_DIR,
                                         'attribute_features',
                                         self.CONFIG)

        self.STORED_FEATURES_DIR = os.path.join(self.FEATURES_DIR,
                                                'all_splits_numpy_format')

        self.PROCESS_FEATURES = args.port_features

        self.EPOCHS = args.epochs
        self.BATCH_SIZE = args.batch_size

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
    # splits (switch from caffe's split name convention to keras's convention)
    X_train = fused_features['train']
    y_train = labels['train']
    X_test = fused_features['eval']
    y_test = labels['eval']
    X_val = fused_features['test']
    y_val = labels['test']

    return X_train, X_val, X_test, y_train, y_val, y_test


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


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def main():
    setup_logging(egosocial.config.LOGGING_CONFIG)

    entry_msg = 'Reproduce experiments in Social Relation Recognition paper.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--project_dir', required=True,
                        help='Base directory.')

    parser.add_argument('--port_features', required=False,
                        action='store_true',
                        help='Whether port features from other formats to'
                             'numpy.')

    parser.add_argument('--reuse_model', required=False,
                        action='store_true',
                        help='Use precomputed model if available.')

    parser.add_argument('--save_model', required=False,
                        action='store_true',
                        help='Save model to disk.')

    parser.add_argument('--save_stats', required=False,
                        action='store_true',
                        help='Save statistics to disk.')

    parser.add_argument('--epochs', required=False, type=positive_int,
                        default=100,
                        help='Max number of epochs.')

    parser.add_argument('--batch_size', required=False, type=positive_int,
                        default=32,
                        help='Batch size.')

    args = parser.parse_args()
    # keep configuration
    conf = Configuration(args)

    train_model(conf)


if __name__ == '__main__':
    main()
