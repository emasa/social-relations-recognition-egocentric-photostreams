# !/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

import numpy as np
import sklearn
from sklearn.decomposition import PCA

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.sequence import pad_sequences


class TimeSeriesArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.
    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        data_generator: Instance of `TimeSeriesDataGenerator`.
        batch_size: Integer, size of a batch.
        maxlen: Length of longest sequence.
        output_cbk: Function called on batch of ys to produce desired output.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        balanced: boolean. Draw samples (with replacement) assuming uniform distribution 
        between classes. Default: false (keras implementation).
    """

    def __init__(self, x, y, data_generator, maxlen=None, output_cbk=None,
                 batch_size=32, shuffle=False, seed=None, balanced=False):
        if y is not None and len(x) != len(y):
            raise ValueError('`x` (images tensor) and `y` (labels) '
                             'should have the same length. '
                             'Found: x.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if maxlen is None:
            maxlen = max(len(seq) for seq in x)

        self.data_generator = data_generator            
        self.maxlen = maxlen
        self.output_cbk = output_cbk
        self.balanced = balanced
        self.n_features = None        

        self.x = []
        for seq in x:
            seq_array = np.asarray(seq, dtype=K.floatx())
            
            if seq_array.ndim != 2:
                raise ValueError('Input data in `TimeSeriesArrayIterator` '
                                 'should have rank 3. You passed an array '
                                 'with shape', seq_array.shape)

            seq_len, seq_n_features = seq_array.shape
                        
            if seq_len > self.maxlen:
                raise ValueError('Input data in `TimeSeriesArrayIterator` '
                                 'should have up to {} timesteps. You passed at least '
                                 'one sample with {} timesteps'.format(self.maxlen, seq_len))

            if self.n_features is None:
                self.n_features = seq_n_features

            if seq_n_features != self.n_features:
                raise ValueError('Input data in `TimeSeriesArrayIterator` '
                                 'should have the same number of features. Found samples with '
                                 '{} != {} features.'.format(self.n_features, seq_n_features))
            
            self.x.append(seq_array)
        
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
            
        if y is not None and balanced:
            classes = np.unique(self.y)
            classes_map = {cls : cls_idx for cls_idx, cls in enumerate(classes)}

            self._y_groupby_cls = [ [] for _ in classes ]                
            for y_idx, cls in enumerate(self.y):
                cls_idx = classes_map[cls]
                self._y_groupby_cls[cls_idx].append(y_idx)
        else:
            self._y_groupby_cls = None

        super(TimeSeriesArrayIterator, self).__init__(len(x), batch_size, shuffle, seed)

    def _set_index_array(self):
        if self.balanced and self.y is not None:
            classes_idx = np.random.randint(0, len(self._y_groupby_cls), size=self.n)
            
            index_array = []
            for cls_idx in classes_idx:
                group = self._y_groupby_cls[cls_idx]
                sample_idx = np.random.randint(0, len(group))
                index_array.append(group[sample_idx])
                
            self.index_array = np.asarray(index_array)
        else:
            super(TimeSeriesArrayIterator, self)._set_index_array()
        
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.data_generator.transform(x.astype(K.floatx()))
            batch_x.append(x)

        batch_x = pad_sequences(batch_x, maxlen=self.maxlen, dtype=K.floatx())

        if self.y is None:
            return batch_x
        
        batch_y = self.y[index_array]
        if self.output_cbk:
            batch_y = self.output_cbk(batch_y)
        
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

class TimeSeriesDataGenerator(object):
    
    def __init__(self, fancy_pca=False, noise_stddev=0.01, random_state=None):
        self.fancy_pca = fancy_pca
        
        self.noise_stddev = noise_stddev
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.eigen_vecs = None
        self.eigen_vals = None
    
    def fit(self, x_sequences):
        if self.fancy_pca:            
            flat_x = np.array(list(itertools.chain(*x_sequences)))

            # compute eigen vectors and eigen values
            pca = PCA(random_state=self.random_state)
            pca.fit(flat_x)

            self.eigen_vecs = pca.components_.T
            self.eigen_vals = pca.explained_variance_
    
    def transform(self, x_seq):        
        if self.fancy_pca:
            if self.eigen_vecs is not None and self.eigen_vals is not None:            
                # for each frame and feature draw gaussian r.v. independently
                alpha = self.random_state.normal(loc=0.0, scale=self.noise_stddev, size=x_seq.shape)
                pca_noise = np.dot(self.eigen_vecs, (alpha * self.eigen_vals).T).T
                # augment data sample
                x_seq = x_seq + pca_noise
            else:
                raise ValueError('Generator needs to be fit before using transform.'
                                 'Call method `.fit(numpy_data)`.')                

        return x_seq
    
    def flow(self, x, y=None, maxlen=None, output_cbk=None, balanced=False,
             batch_size=32, shuffle=True, seed=None):
        """Takes numpy data & label arrays, and generates batches of
            augmented data.
        # Arguments
               x: data. Should have rank 3.
               y: labels.
               maxlen: Length of longest sequence. If None, maxlen is inferred from x (default: None).
               output_cbk: Function called on batch of ys to produce desired output (default: None).
               balanced: boolean. Draw samples (with replacement) assuming uniform distribution 
               between classes. (default: False).               
               batch_size: int (default: 32).
               shuffle: boolean (default: True).
               seed: int (default: None).
        # Returns
            An Iterator yielding tuples of `(x, y)` where `x` is a numpy array of time 
            series data and `y` is a numpy array of corresponding labels."""
        return TimeSeriesArrayIterator(
            x, y, self,
            maxlen=maxlen,
            output_cbk=output_cbk,
            balanced=balanced,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed)