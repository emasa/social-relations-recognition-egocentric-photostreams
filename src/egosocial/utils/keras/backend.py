# !/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import keras.backend as K


def limit_gpu_allocation_tensorflow(memory_percentage=None):

    config = tf.ConfigProto()

    if memory_percentage is not None:
        # fixed amount of gpu memory (across all gpus)
        assert 0 <= memory_percentage <= 1
        config.gpu_options.per_process_gpu_memory_fraction = memory_percentage
    else:
        # on demand
        config.gpu_options.allow_growth = True

    tf_session = tf.Session(config=config)
    K.set_session(tf_session)
