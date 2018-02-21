# !/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import keras.backend as K


def limit_gpu_allocation_tensorflow(memory_percentage=0.25):

    assert 0 <= memory_percentage <= 1

    # default: allocates 25% of total memory
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory_percentage
    tf_session = tf.Session(config=config)
    K.set_session(tf_session)
