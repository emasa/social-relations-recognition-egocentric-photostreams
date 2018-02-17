# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
from os.path import dirname, abspath

sys.path.extend([dirname(dirname(abspath(__file__)))])

import numpy as np
import scipy
import sklearn
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.utils import to_categorical

import egosocial.config
from egosocial.core.types import relation_to_domain, relation_to_domain_vec
from egosocial.utils.caffe.misc import load_levelDB_as_array
from egosocial.utils.keras.autolosses import AutoMultiLossWrapper
from egosocial.utils.logging import setup_logging

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
    # preprocess the data
    data_split = get_data_split(features, selected_attributes,
                                conf.LABEL_FILES, preprocess=True)
    x_train, x_val, x_test, *labels = data_split
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
    clf_wrapper.compile(optimizer='adam',
                        loss=categorical_crossentropy, 
                        loss_weights='auto',
                        #loss_weights=dict(domain=1, relation=2), 
                        metrics=['accuracy'])
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
        #ModelCheckpoint(checkpoint_path, monitor='val_loss',
      #                 save_best_only=True),
        ##TensorBoard(log_dir='./logs', write_images=True, write_graph=True)
    ]

    trainable_model = clf_wrapper.model
    log.info("Training model from scratch...")
    hist = trainable_model.fit(
        x_train_inputs, y_train_outputs,
        batch_size=batch_size, epochs=epochs,
        validation_data=(x_val_inputs, y_val_outputs),
        callbacks=callbacks, verbose=1,
    )


    scores = trainable_model.evaluate(x_test_inputs, y_test_outputs,
                                      batch_size=batch_size)

    log.info(scores)


def get_model(input_shape):
    input_features = Input(shape=input_shape, dtype='float',
                           name='attribute_features')

    from keras import regularizers
    from keras.layers.noise import AlphaDropout
    from keras.layers import Dropout

    x = input_features

#    x = keras.layers.BatchNormalization()(x)

    x = Dense(128, activation='selu', name='dense_1',
                   bias_initializer='lecun_normal',
                   kernel_initializer='lecun_normal',
             )(x)
    x = AlphaDropout(0.25)(x)

    domain = Dense(N_CLS_DOMAIN, activation='softmax', name='domain',
                   bias_regularizer=regularizers.l2(0.1),
                   kernel_regularizer=regularizers.l2(0.1))(x)

    x = keras.layers.concatenate([x, domain])
    relation = Dense(N_CLS_RELATION, activation='softmax', name='relation',
                     bias_regularizer=regularizers.l2(0.1),
                     kernel_regularizer=regularizers.l2(0.1))(x)

    clf = Model(inputs=[input_features], outputs=[domain, relation])

    return clf

def domain_to_relation_matrix():
    W = [np.zeros(N_CLS_RELATION) for _ in range(N_CLS_DOMAIN)]
    for rel in range(N_CLS_RELATION):
        dom = relation_to_domain(rel)
        W[dom] += to_categorical(rel, N_CLS_RELATION)
    return np.array(W).T

def get_model_2(input_shape):
    input_features = Input(shape=input_shape, dtype='float',
                           name='attribute_features')

    x = input_features
    #x = keras.layers.BatchNormalization()(input_features)
    #x = keras.layers.Dropout(0.2)(x)

    from keras import regularizers

    relation = Dense(N_CLS_RELATION, activation='softmax', name='relation',
                     bias_regularizer=regularizers.l2(0.05),
                     kernel_regularizer=regularizers.l2(0.05)
                    )(x)

    weights = [domain_to_relation_matrix()]

    domain = Dense(N_CLS_DOMAIN, activation='linear', name='domain', 
                   use_bias=False, weights=weights, trainable=False)(relation)
    
    clf = Model(inputs=[input_features], outputs=[domain, relation])

    return clf


def preprocess_data(x_train, x_val, x_test, **kwargs):
    n_features = x_train.shape[1]
    # normalize data ?
    normalize = True
    from sklearn.preprocessing import Normalizer, StandardScaler
    if normalize:
        print('Applying normalization to the data')
        #scaler = Normalizer(norm='l2').fit(x_train)
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)
    
    from sklearn.decomposition import PCA
    import pickle

    dim_reduction = True
    min_required, max_required = 200, 200
    min_explained_var = 0.8

    if dim_reduction and n_features > min_required:
        
        quantization = False
        if quantization:
            # Q promotes sparsity, default
            Q = kwargs.get('Q', 32)
            print('Applying Q-sparsity to the data Q={}'.format(Q))
            x_train = np.floor(Q * x_train)
            x_test = np.floor(Q * x_test)
            x_val = np.floor(Q * x_val)
        
        pca_conf = kwargs.get('n_components', min_required)
        if 0 < pca_conf <= 1:
            print('Computing PCA coefficients')
            pca = PCA(n_components=pca_conf)
            pca.fit(x_train)
        else:
            # PCA failed, compute again
            n_components = pca_conf
            for retry in range(3): # 50, 100, 200
                print('Computing PCA retry {}'.format(retry+1))
                pca = PCA(n_components=n_components, svd_solver='randomized')
                pca.fit(x_train)
                
                explained_var = np.sum(pca.explained_variance_ratio_)
                if not np.isnan(explained_var) and explained_var < min_explained_var and n_components < max_required:
                    n_components *= 2
                else:
                    break

        explained_var = np.sum(pca.explained_variance_ratio_)
        n_components = pca.n_components_

        backup_pca = False
        if backup_pca:
            filename = os.path.join(egosocial.config.MODELS_CACHE_DIR, 
                                'pca_exp_var_{}_Q_{}.b'.format(explained_var, Q))
            print('Backing up PCA coefficients in {}.'.format(filename))
            with open(filename, 'wb') as f:
                 pickle.dump(pca, f)

        print('Applying PCA to the data with explained variance {} dims {}'.format(explained_var, n_components))
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        x_val = pca.transform(x_val)
    else:
        print('Skip dimensionality reduction')
 
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


def get_data_split_old(attribute_features, selected_attributes, label_files):
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

def get_data_split(attribute_features, selected_attributes, label_files, 
                   preprocess=False):
    # splits (switch from caffe's split name convention to keras's convention)
    _train, _val, _test = 'train', 'test', 'eval'
    if preprocess:
        from collections import defaultdict
        features = defaultdict(dict)
        # preprocess selected attributes individually
        for attr_name in selected_attributes:
             print('Preprocessing attribute: {}'.format(attr_name))
             data = preprocess_data( attribute_features[_train][attr_name]
                                   , attribute_features[_val][attr_name]
                                   , attribute_features[_test][attr_name])
             for split_idx, split in enumerate((_train, _val, _test)):
                 features[split][attr_name] = data[split_idx]
    else:
        features = attribute_features
    
    # concatenate attributes
    fused_features, labels = {}, {}
    for split in (_train, _val, _test):
        selected_features = [features[split][attr_name]
                             for attr_name in selected_attributes]
    
        fused_features[split] = np.concatenate(selected_features, axis=1)

        with open(label_files[split]) as file_label_list:
            labels[split] = np.array([file_label.split()[1]
                                      for file_label in file_label_list],
                                     dtype=np.int)
    
    result = [fused_features[split] for split in (_train, _val, _test)]
    result.extend([labels[split] for split in (_train, _val, _test)])
    return result


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

    for numpy_file in sorted(os.listdir(features_dir)):
        # Split the extension from the path and normalize it to lowercase.
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

        # some attributes are splitted in two files (one for each person)
        # create a list unique attributes name
        if attr_name.endswith('_1') or attr_name.endswith('_2'):
            attr_name = attr_name[:-2]

        if attr_name in attribute_features[split]:
            array = np.concatenate([attribute_features[split][attr_name], 
                                    features.astype('float32', copy=False)],
                                   axis=1)
        else:
            array = features.astype('float32', copy=False)
        
        attribute_features[split][attr_name] = array

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
