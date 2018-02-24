# !/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import IPython

from keras.callbacks import Callback


class PlotLearning(Callback):

    def __init__(self, figsize=(20, 13), update_step=1):
        super(PlotLearning, self).__init__()
        self.figsize = figsize
        self.update_step = update_step
        # sort legends, training first
        self._legend_key = lambda name: ('val' if name.startswith('val_')
                                               else 'train')

        # internal state
        self.i = None
        self.x = None
        self.metrics = None
        self._metric_map = None
        self._next_idx = None

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []

        self.metrics = defaultdict(list)

        self._metric_map = defaultdict()
        self._metric_map.update(
            {name: idx for idx, name in enumerate(self.model.metrics_names)})
        self._metric_map.update({'val_{}'.format(name): idx for idx, name in
                                 enumerate(self.model.metrics_names)})

        # add other logs if needed (get next available index)
        self._next_idx = lambda: len(set(self._metric_map.values()))
        self._metric_map.default_factory = self._next_idx

        plt.figure(figsize=self.figsize)

    def on_epoch_end(self, epoch, logs={}):
        # update internal state
        self.x.append(self.i+1) # epochs starting in one
        n_plots = 0  # number of unique subplots
        for metric_name in logs.keys():
            # split logs in different lists
            self.metrics[metric_name].append(logs[metric_name])
            # gets the max index (creates new ones if needed)
            n_plots = max(n_plots, self._metric_map[metric_name])
        # indices are zero-based
        n_plots += 1

        # refresh screen every 'update_step' iterations
        if self.i % self.update_step == 0:
            self.plot(n_plots)

        self.i += 1

    def plot(self, n_plots=None):
        # the next index is equal to the number of subplots
        n_plots = n_plots if n_plots else self._next_idx()

        # grid (keeps rectangular shape as compact as possible)
        # when it isn't square shape, horizontal axis is slightly larger than
        # vertical one
        nrows = int(np.floor(np.sqrt(n_plots)))  # vertical axis
        ncols = int(np.ceil(1.0 * n_plots / nrows))  # horizontal axis
        # figure containing multiple plots
        f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)

        # TODO: check if we can get rid of this
        IPython.display.clear_output(wait=True)

        # squeeze array
        ax.shape = (nrows * ncols,)
        # in the same subfigure plot training and validation data for a given
        # metric
        for metric_name in sorted(self.metrics.keys(), key=self._legend_key):
            idx = self._metric_map[metric_name]
            ax[idx].plot(self.x, self.metrics[metric_name], label=metric_name)
            ax[idx].legend()

        plt.show();
