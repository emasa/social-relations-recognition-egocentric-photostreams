# !/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import deque
from multiprocessing.dummy import Pool as ThreadPool

from keras.preprocessing.image import ImageDataGenerator

from ..concurrency import threadsafe_generator
from ..filesystem import check_directory

@threadsafe_generator
def fuse_inputs_generator(generators, inputs, outputs_callback=None): 
    assert len(generators) == len(inputs)
    
    # assume single output (shared by all inputs)
    if outputs_callback is None: # indices 0: first input, 1: label data
        outputs_callback = lambda batch, _inputs: batch[0][1]
    
    n_batches = 3
    n_gens = len(generators)
    tasks = list(generators) * n_batches
    workers = len(tasks)
    
    queue = deque(maxlen=n_batches)
    with ThreadPool(workers) as pool:
        while True:
            if len(queue) == 0:
                batches = pool.map(next, tasks)
                for idx in range(n_batches):
                    data_batch = batches[idx * n_gens : (idx+1) * n_gens]
                    queue.append(data_batch)
            
            data_batch = queue.popleft()            
            # multiple inputs
            X = {input_name : data_batch[idx][0] for idx, input_name in enumerate(inputs)}
            yield X, outputs_callback(data_batch, inputs)


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