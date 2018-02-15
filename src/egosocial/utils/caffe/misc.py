# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import caffe
from caffe.proto import caffe_pb2
import leveldb

def load_levelDB_as_array(db_path):
    db = leveldb.LevelDB(db_path)
    datum = caffe_pb2.Datum()

    items = []

    for key, value in db.RangeIter():
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        items.append(data)

    result = np.array(items).reshape(len(items), len(items[0]))
    return result