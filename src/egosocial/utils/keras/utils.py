# !/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing.dummy import Pool as ThreadPool
import threading

from keras.preprocessing.image import ImageDataGenerator

from ..filesystem import check_directory

class JointGenerator:
    
    def __init__(self, generators, inputs, outputs_callback=None):
        assert len(generators) == len(inputs)
        
        self.generators = generators
        self.inputs = inputs
    
        # assume single output (shared by all inputs)
        if outputs_callback is None: # indices 0: first input, 1: label data
            outputs_callback = lambda batch, _inputs: batch[0][1]
        self.outputs_callback = outputs_callback
        
        self.lock = threading.Lock()

    def _generate_batch(self, generator_index_array):
        generator, index_array = generator_index_array
        return generator._get_batches_of_transformed_samples(index_array)    
    
    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """        
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        # synchronize all sub-generators
        generator_index_array = []        
        with self.lock:
            for image_generator in self.generators:
                with image_generator.lock:
                    index_array = next(image_generator.index_generator)
                    generator_index_array.append((image_generator, index_array))

        # The transformation of images is not under thread lock
        # so it can be done in parallel
        workers = len(self.inputs)
        with ThreadPool(workers) as pool:
            data_batch = pool.map(self._generate_batch, generator_index_array)

        X = {input_name : data_batch[idx][0] for idx, input_name in enumerate(self.inputs)}
        y = self.outputs_callback(data_batch, self.inputs)
        return X, y

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
    
def fuse_inputs_generator(generators, inputs, outputs_callback=None): 
    return JointGenerator(generators, inputs, outputs_callback)

def flow_from_dirs(input_directories, fix_path_cbk=None, **kwargs):
    gen_args = kwargs.pop('gen_args', None)
    gen_args = gen_args if gen_args else {}

    flow_gens = []
    for directory in input_directories:
        check_directory(directory)
        datagen = ImageDataGenerator(**gen_args)
        flow_gen = datagen.flow_from_directory(
            directory, 
            **kwargs
        )

        # fix image links if needed
        # useful when using a symlink to the real image
        if fix_path_cbk:
            flow_gen.filenames = [
                fix_path_cbk(directory, filename) 
                for filename in flow_gen.filenames
            ]

        flow_gens.append(flow_gen)
        
    return flow_gens

import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical

def flow_from_memory(input_directories, fix_path_cbk=None, **kwargs):
    gen_args = kwargs.pop('gen_args', None)
    gen_args = gen_args if gen_args else {}

    flow_gens = []
    for directory in input_directories:
        check_directory(directory)
        datagen = ImageDataGenerator(**gen_args)
        flow_gen = datagen.flow_from_directory(
            directory, 
            **kwargs
        )
        
        # fix image links if needed
        # useful when using a symlink to the real image
        if fix_path_cbk:
            flow_gen.filenames = [
                fix_path_cbk(directory, filename) 
                for filename in flow_gen.filenames
            ]        

        def process_image(fname):
            img = load_img(os.path.join(flow_gen.directory, fname),
                           target_size=flow_gen.target_size,
                           interpolation=flow_gen.interpolation)
            return img_to_array(img, data_format=flow_gen.data_format)
        
        with ThreadPool() as pool:
            xs = pool.map(process_image, flow_gen.filenames)
        
        xs = np.array(xs)
        ys = to_categorical(flow_gen.classes, flow_gen.num_classes)
        
        kwargs_fixed = dict(kwargs)
        invalid_args =  ['target_size', 'color_mode', 'classes', 'class_mode', 'follow_links', 'interpolation']
        for arg in invalid_args:
            kwargs_fixed.pop(arg, None)
        
        datagen = ImageDataGenerator(**gen_args)
        mem_flow_gen = datagen.flow(xs, ys, **kwargs_fixed)
        
        flow_gens.append(mem_flow_gen)
        
    return flow_gens