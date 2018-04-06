# !/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools

from keras.utils import to_categorical
import numpy as np
from sklearn.utils import compute_class_weight

# keep the labels sorted
DOMAINS = ['Attachent', 'Coalitional Group', 'Heirarchical Power', 'Mating', 'Reciprocity']
GROUPED_RELATIONS = [
    [
        'father-child', 'mother-child', 
    #    'grandpa-grandchild', 'grandma-grandchild'
    ],
    [
        #'band members', 'dance team members', 'sport team members', 
        'colleages'
    ],    
    [
        'presenter-audience', 
         #'teacher-student', 'trainer-trainee', 
         'leader-subordinate', 'customer-staff'
    ],
    ['lovers/spouses'],    
    [
        'friends', 
         #'siblings', 
         'classmates'
    ],
]
RELATIONS = sorted(list(itertools.chain(*GROUPED_RELATIONS)))

def relation_to_domain(rel_label):
    for dom_idx, grouped_relations in enumerate(GROUPED_RELATIONS):
        for relation in grouped_relations:
            if rel_label == relation:
                return DOMAINS[dom_idx]

    raise LookupError('Label out of range: {}')

relation_to_domain_vec = np.vectorize(relation_to_domain)

class CategoricalEncoderHelper:
    def __init__(self, categories):
        self.encoding_map = {cls:i for i, cls in enumerate(categories)}        
        self.encode = np.vectorize(self._encode_value)        
    
    def _encode_value(self, cls):
        return self.encoding_map[cls]
        
    def __call__(self, batch_x):
        return to_categorical(self.encode(batch_x), num_classes=len(self.encoding_map))
    
def relation_to_domain_weights():    
    dom_encoder = CategoricalEncoderHelper(DOMAINS)
    rel_encoder = CategoricalEncoderHelper(RELATIONS)

    W = np.zeros((len(DOMAINS), len(RELATIONS)))

    for rel in RELATIONS:
        dom = relation_to_domain(rel)
        dom_idx = np.argmax(dom_encoder(dom))
        W[dom_idx] += rel_encoder(rel)
    
    return W.T

class LabelExpander(object):
    
    def __init__(self, mode=None):
        mode = mode if mode else 'both_splitted'
        assert mode in ('both_splitted', 'relation', 'domain')

        self.mode = mode        
        self.to_categorical_domain = CategoricalEncoderHelper(DOMAINS)
        self.to_categorical_relation = CategoricalEncoderHelper(RELATIONS)
    
    def __call__(self, labels):
        if self.mode == 'both_splitted':
            y_rel = self.to_categorical_relation(labels)
            y_dom = self.to_categorical_domain(relation_to_domain_vec(labels))
            return dict(domain=y_dom, relation=y_rel)
        elif self.mode == 'domain':
            # assume domain-specific labels
            y_dom = self.to_categorical_domain(labels)            
            return dict(domain=y_dom)
        else:
            y_rel = self.to_categorical_relation(labels)            
            return dict(relation=y_rel)

def decode_prediction(prediction, mode=None):
    mode = mode if mode else 'both_splitted'
    assert mode in ('both_splitted', 'relation', 'domain')    

    decoded_output = {}
    if mode == 'both_splitted':
        decoded_output['domain'] = prediction[0]
        decoded_output['relation'] =  prediction[1]        
    elif mode == 'domain':
        decoded_output['domain'] =  prediction
    elif mode == 'relation':
        decoded_output['relation'] =  prediction        

    if 'domain' in decoded_output:
        domain_idx = np.argmax(decoded_output['domain'], axis=-1)
        domains = np.asarray(DOMAINS)       
        decoded_output['domain'] = domains[domain_idx]

    if 'relation' in decoded_output:
        relation_idx = np.argmax(decoded_output['relation'], axis=-1)
        relations = np.asarray(RELATIONS)
        decoded_output['relation'] = relations[relation_idx]

    return decoded_output

def compute_class_weight_labels(y, mode=None):
    if mode == 'both_fused':
        raise NotImplementedError('Class weights are not available in mode both_fused.')

    encoder = LabelExpander(mode=mode)
    y = encoder(y)

    class_weight = {}
    for y_type, y_data in y.items():
        data = list(np.argmax(y_data, axis=1))    
        classes = np.unique(data)
        weights = compute_class_weight('balanced', classes, data)
        class_weight[y_type] = dict(zip(classes, weights))
    
    return class_weight