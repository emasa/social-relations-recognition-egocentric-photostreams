# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy

class StratifiedGroupShuffleSplitWrapper(object):
    
    def __init__(self, splitter, n_splits=10, max_test_size=0.25, min_test_size=0.15):
        self._split_impl = splitter
        self.n_splits = n_splits
        self.max_test_size = max_test_size
        self.min_test_size = min_test_size
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def split(self, X, y=None, groups=None, return_score=False):
        n_classes = len(np.unique(y))
        
        valid_splits = []
        for train_index, test_index in self._split_impl.split(X, y, groups):
            train_y, test_y = y[train_index], y[test_index]            
            test_ratio = len(test_y) / (len(train_y) + len(test_y))
            
            if not (self.min_test_size <= test_ratio <= self.max_test_size):
                continue
            
            u_train_y, train_counts = np.unique(train_y, return_counts=True)
            u_test_y, test_counts = np.unique(test_y, return_counts=True)
            
            if len(u_train_y) == n_classes and len(u_test_y) == n_classes:
                train_freq, test_freq = 1. * train_counts / len(train_y), 1. * test_counts / len(test_y)
                
                #cls_freq_error = np.linalg.norm(train_freq-test_freq)
                cls_freq_error = scipy.stats.entropy(train_freq, test_freq)
                
                valid_splits.append([train_index, test_index, cls_freq_error])

        if len(valid_splits) < self.n_splits:
            raise ValueError("Couldn't compute {} splits with the given criteria. " 
                             "Found {} splits.".format(self.n_splits, len(valid_splits)))
        
        valid_splits = sorted(valid_splits, key=lambda x : x[-1])
        for idx in range(self.n_splits):
            train_index, test_index, cls_freq_error = valid_splits[idx]
            if return_score:
                yield train_index, test_index, cls_freq_error
            else:
                yield train_index, test_index
