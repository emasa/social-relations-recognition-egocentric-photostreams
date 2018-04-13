# !/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
import functools
import itertools
import json
import logging
import math
import os
import pprint
import sys
import pickle
import datetime
from shutil import rmtree

sys.path.extend([os.path.dirname(os.path.abspath('.'))])

import numpy as np
import pandas as pd
import sklearn
import scipy

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

import keras
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

import egosocial
import egosocial.config
from egosocial.core.attributes import AttributeSelector
from egosocial.core.models import create_model_top_down
from egosocial.core.models import create_model_bottom_up
from egosocial.core.models import create_model_independent_outputs
from egosocial.utils.filesystem import create_directory 
from egosocial.utils.filesystem import check_directory
from egosocial.utils.keras.autolosses import AutoMultiLossWrapper
from egosocial.utils.keras.backend import limit_gpu_allocation_tensorflow
from egosocial.utils.keras.callbacks import PlotLearning
from egosocial.utils.keras.metrics import precision
from egosocial.utils.keras.metrics import recall
from egosocial.utils.keras.metrics import fmeasure
from egosocial.utils.keras.processing import TimeSeriesDataGenerator
from egosocial.utils.logging import setup_logging
from egosocial.utils.misc import RELATIONS, DOMAINS
from egosocial.utils.misc import LabelExpander
from egosocial.utils.misc import relation_to_domain_vec
from egosocial.utils.sklearn.model_selection import StratifiedGroupShuffleSplitWrapper
from egosocial.utils.keras.scikit_learn import KerasGeneratorClassifier

SHARED_SEED = 42

# # Helper functions

def parse_day(image_path):
    image_name = os.path.basename(image_path)
    # a valid image follows the day_hour_x.ext format
    day_hour_rest = image_name.split('_')
    
    if len(day_hour_rest) == 3:
        # day is the first item
        return day_hour_rest[0]
    else:
        # day isn't available
        return ''
    
def load_dataset_defition(dataset_path, include_day=True):
    with open(dataset_path, 'r') as json_file:
        dataset_def = json.load(json_file)

    # flatten the segments structure
    samples = pd.DataFrame(list(itertools.chain(*dataset_def)))
    
    if include_day:
        samples['day'] = samples['global_image_path'].apply(parse_day)
    
    return samples

def as_sequences(features, sequences_index):

    feature_sequences = []
    for index in sequences_index:
        feature_seq = features[index]
        feature_seq.shape = (feature_seq.shape[0], -1)
        feature_sequences.append(feature_seq)
    
    return np.asarray(feature_sequences)

def load_features(features_path, sequences_index):    
    return as_sequences(np.load(features_path), sequences_index)

def load_fields(data_frames, fields, valid_frames_idx=None):
    assert len(fields) > 0
    
    if valid_frames_idx is None:
        sequences_info = data_frames.groupby(['split', 'segment_id', 'group_id'])
    else:
        sequences_info = data_frames[valid_frames_idx].groupby(['split', 'segment_id', 'group_id'])
    
    fst_seq_frames = [group.index[0] for _, group in sequences_info]
    fields_data =  data_frames.iloc[fst_seq_frames][fields].values

    return [fields_data[:, field_idx] for field_idx in range(len(fields))]        
                
class DimReductionTransformer(object):        

    def __init__(self, n_components, Q=32, normalize=True, random_state=None):
        # PCA configuration (number of components or min explained variance)
        self.pca_param = n_components
         # features quantization (smaller Q promotes sparsity)
        self.Q = Q
        self.normalize = normalize
        self.random_state = sklearn.utils.check_random_state(random_state)
        
        self._scaler = None
        self._pca = None
        
        self._log = logging.getLogger(self.__class__.__name__)        
                
    def fit(self, x):
        # reset state
        self._scaler = self._pca = None        
    
        if self.normalize:
            if self.Q: # quantization requires data in range [0, 1] 
                self._scaler = Normalizer(norm='l2')
            else:
                self._scaler = StandardScaler()
            
            x = self._scaler.fit_transform(x)

        if self.Q:
            # small Q promotes sparsity
            x = np.floor(self.Q * x)            
            
        assert self.pca_param > 0     
        # compute pca from scratch
        if 0 < self.pca_param <= 1:
            # running pca with min explained variance takes much longer
            self._pca = PCA(self.pca_param, random_state=self.random_state)
        else:
            self._pca = PCA(self.pca_param, svd_solver='randomized', random_state=self.random_state)
        
        self._pca.fit(x)
    
    def transform(self, x):
        if self.normalize:
            x = self._scaler.transform(x)

        if self.Q:
            # small Q promotes sparsity
            x = np.floor(self.Q * x)            

        # pca transformation
        x = self._pca.transform(x)

        return x

class Preprocessing(BaseEstimator):
    
    def __init__(self, features_range=None, create_transformation_cbk=None):
        self.features_range = features_range if features_range else dict(all=(0,-1))
        self.create_transformation_cbk = create_transformation_cbk
        self._transformation_dict = {}
        
    def fit(self, X, y=None):
        self._transformation_dict = {}
        
        X = np.concatenate(list(itertools.chain(X)))     
        for features_id in sorted(self.features_range.keys()):
            transformation = self._transformation_dict[features_id] = self.create_transformation_cbk(features_id)
            if transformation:
                begin_slice, end_slice = self.features_range[features_id] 
                X_features = X[:, begin_slice:end_slice]
                transformation.fit(X_features)
        
        return self
    
    def transform(self, X):
        
        seq_length = list(map(len, X))
        X = np.concatenate(list(itertools.chain(X)), axis=0)
        
        features_list = []
        for features_id in sorted(self.features_range.keys()):
            transformation = self._transformation_dict.get(features_id, None)            

            begin_slice, end_slice = self.features_range[features_id] 
            X_features = X[:, begin_slice:end_slice]
            
            if transformation:
                X_features = transformation.transform(X_features)
            
            features_list.append(X_features)
            
        transformed_features = np.concatenate(features_list, axis=-1)            
        seq_end = list(np.cumsum(seq_length))
        seq_begin = [0] + seq_end[:-1]
        sequences = np.asarray([transformed_features[begin:end, :] for begin, end in zip(seq_begin, seq_end)])
        
        return sequences

class TransformationFactory(object):

    def __init__(self, n_components=50, Q=32, seed=None):
        self.n_components = n_components
        self.Q = Q
        self.seed = seed

    def __call__(self, attribute):
        if attribute in ('camera_user_age', 'camera_user_gender'):
            return None
        elif attribute == 'distance':
            return MinMaxScaler()
        else:
            return DimReductionTransformer(n_components=self.n_components, Q=self.Q, random_state=self.seed)    
        
# Utility function to report best scores
def report(results, n_top=3, metric='score'):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_' + metric] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation {}: {:.3f} (std: {:.3f})".format(
                  metric,
                  results['mean_test_' + metric][candidate],
                  results['std_test_' + metric][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
# # Main class
class SocialClassifierWithPreComputedFeatures:
    
    def __init__(self, dataset_path, features_dir, test_size=0.2, k_fold_splits=10, val_size=None, n_components=50, Q=32, seed=42):
        self.dataset_path = dataset_path
        self.features_dir = features_dir
        self.test_size = test_size
        self.k_fold_splits = k_fold_splits
        self.val_size = val_size if val_size else 1.0 / k_fold_splits
        self.seed = seed

        self._max_seq_len = None
        self._labels = None
        self._users = None

        self.attributes = []
        self.features = None
        self.features_range = None

        self._train_idx = None
        self._test_idx = None
        self._k_train_val_idx = None
        
        self._log = logging.getLogger(self.__class__.__name__)

    def load_data(self):        
        # load dataset definition
        frames = load_dataset_defition(self.dataset_path, include_day=True)
        # filter labels with few samples
        valid_frames_idx = np.isin(frames['relation_label'], RELATIONS)

        # for each sequence get label, user, day of first frame
        self._labels, self._users, self._seq_days = load_fields(
            frames, ['relation_label', 'camera_user_name', 'day'], 
            valid_frames_idx=valid_frames_idx
        )
                
        grouped_frames = frames[valid_frames_idx].groupby(['split', 'segment_id', 'group_id'])        
        self._max_seq_len = grouped_frames.size().max()
        
        sequences_index = [list(group.index) for _, group in grouped_frames]
        self._init_features(sequences_index)

        self._init_grouped_splits()

    def _init_features(self, sequences_index):
        attribute_files = sorted(next(os.walk(self.features_dir))[2])
        
        ext = '.npy'
        self.attributes = [os.path.splitext(file)[0] for file in attribute_files if file.endswith(ext)]
        self.features_range = {}

        # load features
        attribute_list = []
        begin = 0
        for attribute_name in self.attributes:
            path = os.path.join(self.features_dir, attribute_name + ext)
            attribute_features = np.load(path)
            attribute_features.shape = (attribute_features.shape[0], -1)
            self._log.debug('Loading features {} dim: {}'.format(attribute_name, attribute_features.shape))
            end = begin + attribute_features.shape[-1]            

            attribute_list.append(attribute_features)            
            self.features_range[attribute_name] = (begin, end)            
            begin = end
        
        self.features = as_sequences(np.concatenate(attribute_list, axis=-1), sequences_index)
        
    def _init_grouped_splits(self):
        # define data splits
        # define train, test splits        
        criteria = np.array([ user + '_' + day for user, day in zip(self._users, self._seq_days) ])
        y, groups = self._labels, criteria

        n_tries, group_size, epsilon = 1000, self.test_size, 0.025

        split_wrapper = StratifiedGroupShuffleSplitWrapper(
            GroupShuffleSplit(n_splits=n_tries, test_size=group_size, random_state=self.seed), 
            n_splits=1, 
            max_test_size=min(self.test_size + epsilon, 1.0), min_test_size=max(self.test_size - epsilon, 0.0)
        )
        self._train_idx, self._test_idx, train_test_score = next(split_wrapper.split(np.zeros(len(y)), y, groups, return_score=True))
        test_size = len(self._test_idx) / (len(self._train_idx) + len(self._test_idx))
        self._log.debug('Split train-test score: {:.3} real_test_size: {:.3}'.format(train_test_score, test_size))
        
        # define k-fold splits
        y, groups = y[self._train_idx], groups[self._train_idx]
        
        if self.k_fold_splits > 1:
            # k-fold strategy
            # search x times the number of splits, encourage diversity
            # double the epsilon (more flexible)
            n_tries, group_size, epsilon = self.k_fold_splits * 50, self.val_size, 0.05
        else:
            # holdout strategy
            n_tries, group_size, epsilon = 1000, self.val_size, 0.05
        
        split_wrapper = StratifiedGroupShuffleSplitWrapper(
            GroupShuffleSplit(n_splits=n_tries, test_size=group_size, random_state=self.seed), 
            n_splits=self.k_fold_splits,
            max_test_size=min(self.val_size + epsilon, 1.0), min_test_size=max(self.val_size - epsilon, 0.0)
        )
        
        self._k_train_val_idx = []
        for k, (t_idx, v_idx, t_v_score) in enumerate(split_wrapper.split(np.zeros(len(y)), y, groups, return_score=True)):
            self._k_train_val_idx.append((self._train_idx[t_idx], self._train_idx[v_idx]))
            val_size = (1 - test_size) * len(v_idx) / (len(t_idx) + len(v_idx))
            self._log.debug('{}-fold split score: {:.3} real_val_size={:.3}'.format(k, t_v_score, val_size))            
            
    def list_attributes(self):
        # list attributes
        return self.attributes
            
    def get_split_idx(self, split, k_fold=None):
        assert split in ('train', 'test', 'val')
        if split == 'train':
            if k_fold is None:
                return self._train_idx
            else:
                assert 0 <= k_fold < self.k_fold_splits
                return self._k_train_val_idx[k_fold][0]

        if split == 'val':
            assert 0 <= k_fold < self.k_fold_splits
            return self._k_train_val_idx[k_fold][1]
        
        if split == 'test':
            return self._test_idx

    def max_sequence_len(self):
        return self._max_seq_len


# # Configure Keras

def init_callbacks(output_mode, plot_stats=True, save_model=False, save_stats=False, stop_early=False, plot_step=1, reduce_lr=False, figsize=None):
    callbacks = []

    training_dir = os.path.join(egosocial.config.TMP_DIR, 'training')
    create_directory(training_dir, 'Training')

    if save_model:
        checkpoint_path = os.path.join(training_dir,
                                       'weights.{epoch:02d}-{val_loss:.2f}.h5')
        checkpointer = ModelCheckpoint( 
            filepath=checkpoint_path, monitor='val_loss',
            save_best_only=True, period=5,
        )
        callbacks.append(checkpointer)

    if save_stats:
        metrics_path = os.path.join(training_dir,
                                    'metrics.csv')
        csv_logger = CSVLogger(metrics_path)
        callbacks.append(csv_logger)

    if reduce_lr:
        lr_handler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
        )
        callbacks.append(lr_handler)
   
    if plot_stats:
        # more plots need more space
        if not figsize:
            if output_mode != 'both_splitted':
                figsize = (25, 5)
            else:
                figsize = (25, 13)

#        plot_metrics = PlotLearning(update_step=plot_step, figsize=figsize)
#        callbacks.append(plot_metrics)
        
    if stop_early:
        stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')
        callbacks.append(stopper)
    
    return callbacks

def compile_model(
    model, 
    optimizer='adam', 
    loss='categorical_crossentropy', 
    loss_weights='auto',
    **kwargs
):
    # wrapper allows to train the loss weights
    model_wrapper = AutoMultiLossWrapper(model)
    model_wrapper.compile(optimizer=optimizer, loss=loss, 
                          loss_weights=loss_weights, **kwargs)

    return model_wrapper.model

def build_model(
    mode='both_splitted',
    model_strategy='top_down', 
    recurrent_type='LSTM',
    max_seq_len=34,
    feature_vector_size=None,
    hidden_fc=0,
    units=128, 
    drop_rate=0.5,
    l2_reg=0.01,    
    learning_rate=0.0001,
    loss='categorical_crossentropy',    
    loss_weights='auto',
    metrics=None,
    decay=1e-5,
):
    if feature_vector_size is None:
        global n_features
        feature_vector_size = n_features
    
    model_strategy_select = {
        'top_down' : create_model_top_down,
        'bottom_up' : create_model_bottom_up,
        'independent' : create_model_independent_outputs,
    }
    
    model_parameters = dict(
        mode=mode,
        max_seq_len=max_seq_len, 
        n_features=feature_vector_size,
        units=units,
        drop_rate=drop_rate,
        rec_drop_rate=drop_rate,        
        l2_reg=l2_reg,
        hidden_fc=hidden_fc,
        recurrent_type=recurrent_type,
        n_relations=len(RELATIONS),
        n_domains=len(DOMAINS),
        seed=SHARED_SEED,
    )
    
    model = model_strategy_select[model_strategy](**model_parameters)
    if model_strategy_select == 'bottom_up':
        model.get_layer('domain').set_weights([relation_to_domain_weights()])
        
    model = compile_model(
        model,   
        optimizer=keras.optimizers.Adam(learning_rate, decay=decay),
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics if metrics else ['accuracy'],
    )
    
    return model

class TimeSeriesDataGeneratorBuilder(object):
    def __init__(self, noise_stddev=0.01, seed=SHARED_SEED, maxlen=None, output_cbk=None, balanced=False):
        self.noise_stddev = 0.01
        self.seed = seed
        self.maxlen = maxlen
        self.output_cbk = output_cbk
        self.balanced = balanced

    def __call__(self, X, y=None, batch_size=32, phase='train'):
        assert phase in ('train', 'test')  
        
        if phase == 'train':
            datagen = TimeSeriesDataGenerator(fancy_pca=True, 
                                              noise_stddev=self.noise_stddev, 
                                              random_state=self.seed)
            datagen.fit(X)
        else:
            datagen = TimeSeriesDataGenerator(fancy_pca=False)
        
        shuffle = (phase == 'train')
        
        if phase == 'train':
            generator = datagen.flow(
                X, y,
                maxlen=self.maxlen,
                output_cbk=self.output_cbk,
                balanced=self.balanced,
                batch_size=batch_size,
                seed=self.seed,
                shuffle=shuffle,
            )
        else:
            generator = datagen.flow(
                X, y,
                maxlen=self.maxlen,
                output_cbk=self.output_cbk,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            
        return generator

    # learning rate schedule
class StepDecay(object):

    def __init__(self, initial_lr=0.01, drop_rate=0.5, epochs_drop=10.0):
        self.initial_lr = initial_lr
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop

    def __call__(self, epoch):
        lrate = self.initial_lr * math.pow(self.drop_rate, math.floor((1+epoch)/self.epochs_drop))
        return lrate
    
def run(conf):    
    # # Loading precomputed features and labels
    helper = SocialClassifierWithPreComputedFeatures(
        conf.dataset_path, conf.features_dir, 
        test_size=0.2, 
        k_fold_splits=3,
        val_size=0.2, # relative to training size
        seed=SHARED_SEED
    )

    helper.load_data()

    # # Prepare splits
    train_val_splits = [ 
        (helper.get_split_idx('train', k_fold=k), helper.get_split_idx('val', k_fold=k)) 
        for k in range(helper.k_fold_splits)
    ]

    # # Parameters
    n_components, Q = conf.pca_components, conf.Q
    n_features = n_components * 9 + 6 + 2 + 1
    max_timestep = helper.max_sequence_len()    
    
    helper._log.info('Number of pca components per attribute (for visual embeddings): {}'.format(n_components))
    helper._log.info('Q: {}'.format(Q))
    helper._log.info('Length of the largest sequence: {}'.format(max_timestep))
    helper._log.info('Total number of features: {}'.format(n_features))

    #'both_splitted' # multi-loss domain-relation
    #'domain' # domain only
    #'relation' # relation only
    output_mode = conf.output_mode
    helper._log.info('Output mode: {}'.format(output_mode))

    model_strategy = conf.model_strategy
    helper._log.info('Model strategy: {}'.format(model_strategy))
    
    # # Grid search CV
    reduce_dim = Preprocessing(
        features_range=helper.features_range, 
        create_transformation_cbk=TransformationFactory(n_components=n_components, Q=Q, seed=SHARED_SEED)
    )
    generator_builder = TimeSeriesDataGeneratorBuilder(
        maxlen=max_timestep,
        output_cbk=LabelExpander(mode=output_mode),
        seed=SHARED_SEED,
    )

    single_output = conf.single_output
    metric_suffix = 'fmeasure'
    
    # used only if GridSearchCV scoring attribute is set to None
    if output_mode in ('domain', 'relation'):
        metric_score = metric_suffix
    else:
        metric_score = single_output + '_' + metric_suffix

    clf = KerasGeneratorClassifier(
        build_fn=build_model,
        build_generator=generator_builder,
        output_mode=output_mode,
        metric_score=metric_score,
        single_output=single_output,
        balanced=True,        

        max_seq_len=max_timestep,
        feature_vector_size=n_features,        
        recurrent_type='LSTM',
        hidden_fc=1,
        mode=output_mode,
        model_strategy=model_strategy,
        
        metrics=['accuracy', fmeasure],
        verbose=1,
        workers=2,
    )

    pipeline = Pipeline([('reduce_dim', reduce_dim), ('clf', clf)])

    common_search_params = dict(
        estimator=pipeline, 
        cv=train_val_splits,
        scoring=
        ['accuracy', 
         'recall_weighted', 'precision_weighted', 'f1_weighted', 
         'recall_macro', 'precision_macro', 'f1_macro'],
        refit=False,
        return_train_score=True,
        iid=False,
        verbose=4,
        n_jobs=1,
    )

    param_grid = dict(
        clf__drop_rate=conf.drop_rate,
        clf__learning_rate=conf.lr,
        clf__epochs=conf.epochs,
        clf__units=conf.units,
        clf__l2_reg=conf.l2_reg,
        clf__batch_size=conf.batch_size,
        clf__decay=conf.lr_decay, 
    )

    
    do_search = 'grid'    
    search_cv = GridSearchCV(
        param_grid=param_grid,
        **common_search_params,
    )
    
    callbacks = []
    if conf.schedule_lr:
        learning_rate = conf.lr[0]
        lr_scheduler = LearningRateScheduler(StepDecay(initial_lr=learning_rate, drop_rate=0.5, epochs_drop=10))
        callbacks.append(lr_scheduler)
    
    X = helper.features
    if output_mode == 'domain':
        # domain specific-labels
        y = relation_to_domain_vec(helper._labels)
    else:
        y = helper._labels

        
    fit_params = dict(
        clf__verbose=1,
        clf__callbacks=callbacks,
    )
        
    search_result = search_cv.fit(X, y, **fit_params)

    training_dir = os.path.join(egosocial.config.TMP_DIR, 'training')
    create_directory(training_dir, 'Training')

    date_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    file_name = '{}_{}_results_{}_{}.pkl'.format(date_str, do_search, output_mode, model_strategy)
    results_path = os.path.join(training_dir, file_name)

    with open(results_path, 'wb') as file:
        pickle.dump(search_result.cv_results_, file, protocol=pickle.HIGHEST_PROTOCOL)

def parse_list_arg(list_arg, parse_cbk=str, sep=','):
    return list(map(parse_cbk, list_arg.strip().split(sep)))
    
def main():
    entry_msg = 'Grid search for training model for social relations classification in egosocial photo-streams.'
    parser = argparse.ArgumentParser(description=entry_msg)

    parser.add_argument('--dataset_path', required=True,
                       help='Path to file containing the input data and labels information merged.')

    parser.add_argument('--features_dir', required=True,
                       help='Directory where the extracted features are stored.')

    parser.add_argument('--batch_size', required=False,
                       default='64',
                       help='Batch size. Default: 64. Accept multiple values separated by , (comma).')
    
    parser.add_argument('--epochs', required=False,
                        default='100',
                        help='Max number of epochs. Default: 100. Accept multiple values separated by , (comma).')

    parser.add_argument('--lr', required=False,
                        default='0.001',
                        help='Initial learning rate. Default: 0.001. Accept multiple values separated by , (comma).')

    parser.add_argument('--drop_rate', required=False,
                        default='0.5',
                        help='Dropout rate. Default: 0.5. Accept multiple values separated by , (comma).')    

    parser.add_argument('--l2_reg', required=False,
                        default='0.001',
                        help='L2 regularizarion term. Default: 0.001. Accept multiple values separated by , (comma).')    
    
    parser.add_argument('--units', required=False,
                        default='128',
                        help='Number of neurons in lstm/fc hidden layers. Default: 128. Accept multiple values separated by , (comma).')
    
    parser.add_argument('--Q', required=False, type=int,
                        default=32,
                        help='Q discretization parameter. Default: 32.')
    
    parser.add_argument('--pca_components', required=False, type=int,
                        default=50,
                        help='Number of pca components used for visual embedding dim reduction. Default: 50.')    
    
    parser.add_argument('--output_mode', required=False,
                        default='both_splitted',
                        choices=['both_splitted', 'domain', 'relation'],
                        help='Model output. Default: both_splitted, uses multi-loss learning.')
    
    parser.add_argument('--model_strategy', required=False,
                        default='top_down',
                        choices=['top_down', 'bottom_up', 'independent'],
                        help='Model strategy. Default: top_down. Set output_mode=both_splitted for multi-loss learning.')
    
    parser.add_argument('--single_output', required=False,
                        default='relation',
                        choices=['relation', 'domain'],
                        help='Metrics can be computed on a single output during grid-search. Default: relation.')

    parser.add_argument('--schedule_lr', required=False,
                        action='store_true',
                        help='Whether use learning rate schedule. If enabled, --lr must be unique.')
    
    parser.add_argument('--lr_decay', required=False,
                        default='1e-5',
                        help='Learning rate decay on every update. Default: 1e-5. Accept multiple values separated by , (comma).')  
    
    conf = parser.parse_args()
    conf.batch_size = parse_list_arg(conf.batch_size, parse_cbk=int)
    conf.epochs = parse_list_arg(conf.epochs, parse_cbk=int)
    conf.units = parse_list_arg(conf.units, parse_cbk=int)
    conf.lr = parse_list_arg(conf.lr, parse_cbk=float)
    conf.l2_reg = parse_list_arg(conf.l2_reg, parse_cbk=float)
    conf.drop_rate = parse_list_arg(conf.drop_rate, parse_cbk=float)
    conf.lr_decay = parse_list_arg(conf.lr_decay, parse_cbk=float)    
    
    if conf.schedule_lr:
        assert conf.schedule_lr and len(conf.lr) == 1
    
    if not os.path.isdir(egosocial.config.TMP_DIR):
        os.mkdir(egosocial.config.TMP_DIR)

    setup_logging(egosocial.config.LOGGING_CONFIG,
                 log_dir=egosocial.config.LOGS_DIR)
    
    # # Limit GPU memory allocation with Tensorflow
    limit_memory = True
    if limit_memory and K.backend() == 'tensorflow':
        memory_ratio = 0.3
        limit_gpu_allocation_tensorflow(memory_ratio)   
        
    run(conf)

if __name__ == '__main__':
    main()
