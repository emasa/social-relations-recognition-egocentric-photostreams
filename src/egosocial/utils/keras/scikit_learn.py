# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import types

import numpy as np

from keras.utils.generic_utils import has_arg
from keras.models import Sequential

from ..misc import compute_class_weight_labels
from ..misc import decode_prediction

class BaseGeneratorWrapper(object):
    """Base class for the Keras scikit-learn wrapper.
    Warning: This class should not be used directly.
    Use descendant classes instead.
    # Arguments
        build_fn: callable function or class instance
        **sk_params: model parameters & fitting parameters
    The `build_fn` should construct, compile and return a Keras model, which
    will then be used to fit/predict. One of the following
    three values could be passed to `build_fn`:
    1. A function
    2. An instance of a class that implements the `__call__` method
    3. None. This means you implement a class that inherits from either
    `KerasClassifier` or `KerasRegressor`. The `__call__` method of the
    present class will then be treated as the default `build_fn`.
    `sk_params` takes both model parameters and fitting parameters. Legal model
    parameters are the arguments of `build_fn`. Note that like all other
    estimators in scikit-learn, `build_fn` should provide default values for
    its arguments, so that you could create the estimator without passing any
    values to `sk_params`.
    `sk_params` could also accept parameters for calling `fit`, `predict`,
    `predict_proba`, and `score` methods (e.g., `epochs`, `batch_size`).
    fitting (predicting) parameters are selected in the following order:
    1. Values passed to the dictionary arguments of
    `fit`, `predict`, `predict_proba`, and `score` methods
    2. Values passed to `sk_params`
    3. The default values of the `keras.models.Sequential`
    `fit`, `predict`, `predict_proba` and `score` methods
    When using scikit-learn's `grid_search` API, legal tunable parameters are
    those you could pass to `sk_params`, including fitting parameters.
    In other words, you could use `grid_search` to search for the best
    `batch_size` or `epochs` as well as the model parameters.
    """

    def __init__(self, build_fn=None, **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.check_params(sk_params)

    def check_params(self, params):
        """Checks for user typos in `params`.
        # Arguments
            params: dictionary; the parameters to be checked
        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        legal_params_fns = [Sequential.fit_generator, 
                            Sequential.predict_generator,
                            Sequential.evaluate_generator,
                           ]
        if self.build_fn is None:
            legal_params_fns.append(self.__call__)
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            legal_params_fns.append(self.build_fn.__call__)
        else:
            legal_params_fns.append(self.build_fn)

        for params_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, params_name):
                    break
            else:
                if params_name != 'nb_epoch' and params_name != 'batch_size':
                    raise ValueError(
                        '{} is not a legal parameter'.format(params_name))

    def get_params(self, **params):
        """Gets parameters for this estimator.
        # Arguments
            **params: ignored (exists for API compatibility).
        # Returns
            Dictionary of parameter names mapped to their values.
        """
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        """Sets the parameters of this estimator.
        # Arguments
            **params: Dictionary of parameter names mapped to their values.
        # Returns
            self
        """
        self.check_params(params)
        self.sk_params.update(params)
        return self

    def filter_sk_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.
        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`
        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.sk_params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res

    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        """
        raise NotImplementedError("´fit´ method not implemented.")

class KerasGeneratorClassifier(BaseGeneratorWrapper):

    def __init__(self, 
                 build_fn=None, 
                 build_generator=None,
                 metric_score='acc',
                 single_output=None,
                 balanced=False,
                 output_mode=None,
                 **sk_params):
        super(KerasGeneratorClassifier, self).__init__(build_fn=build_fn, **sk_params)
        assert build_generator is not None
        self.build_generator = build_generator
        self.metric_score = metric_score
        self.single_output = single_output
        self.balanced = balanced
        self.output_mode = output_mode

    def get_params(self, **params):
        """Gets parameters for this estimator.
        # Arguments
            **params: ignored (exists for API compatibility).
        # Returns
            Dictionary of parameter names mapped to their values.
        """
        res = super(KerasGeneratorClassifier, self).get_params(**params)
        res.update({
            'build_generator': self.build_generator,
            'metric_score': self.metric_score,
            'single_output': self.single_output,
            'balanced': self.balanced,
            'output_mode': self.output_mode,
        })
        return res

    def set_params(self, **params):
        """Sets the parameters of this estimator.
        # Arguments
            **params: Dictionary of parameter names mapped to their values.
        # Returns
            self
        """
        if 'build_generator' in params:
            self.build_generator = params.pop('build_generator')
        if 'metric_score' in params:
            self.metric_score = params.pop('metric_score')
        if 'single_output' in params:
            self.single_output = params.pop('single_output')
        if 'balanced' in params:
            self.balanced = params.pop('balanced')
        if 'output_mode' in params:
            self.output_mode = params.pop('output_mode')
            
        self.check_params(params)
        self.sk_params.update(params)
        return self
                
    def fit(self, x, y, sample_weight=None, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.
        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`
        # Returns
            history : object
                details about the training history at each epoch.
        # Raises
            ValueError: In case of invalid shape for `y` argument.
        """
        y = np.asarray(y)
        
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(
                **self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))
        
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)

        batch_size = self.sk_params.get("batch_size", 32)
        steps = fit_args.pop("steps_per_epoch", None)

        if steps is None:
            steps = int(np.ceil(x.shape[0] / batch_size))

        if self.balanced:
            # class_weight for keras (balance domain/relation instances)
            class_weight = compute_class_weight_labels(y, mode=self.output_mode)
        else:
            class_weight = None
        
        return self.model.fit_generator(
            self.build_generator(x, y, batch_size=batch_size, phase='train'),
            steps_per_epoch=steps,
            class_weight=class_weight,
            **fit_args
        )

    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.
        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """           
        predict_args = self.filter_sk_params(Sequential.predict_generator, kwargs)
        predict_args.update(kwargs)
        
        batch_size = self.sk_params.get("batch_size", 32)
        steps = predict_args.pop("steps", None)
            
        if steps is None:
            if x.shape[0] % batch_size != 0:
                batch_size, steps = x.shape[0], 1
            else:
                steps = int(x.shape[0] / batch_size)
        
        proba = self.model.predict_generator(
            self.build_generator(x, batch_size=batch_size, phase='test'),
            steps=steps, 
            **predict_args,
        )
        
        classes = decode_prediction(proba, mode=self.output_mode)
        
        if self.single_output:
            classes = classes[self.single_output]
            
        return classes

    def predict_proba(self, x, **kwargs):
        """Returns class probability estimates for the given test data.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.
        # Returns
            proba: array-like, shape `(n_samples, n_outputs)`
                Class probability estimates.
                In the case of binary classification,
                to match the scikit-learn API,
                will return an array of shape `(n_samples, 2)`
                (instead of `(n_sample, 1)` as in Keras).
        """
        predict_args = self.filter_sk_params(Sequential.predict_generator, kwargs)
        predict_args.update(kwargs)

        batch_size = self.sk_params.get("batch_size", 32)
        steps = predict_args.pop("steps", None)

        if steps is None:
            if x.shape[0] % batch_size != 0:
                batch_size, steps = x.shape[0], 1
            else:
                steps = int(x.shape[0] / batch_size)

        probs_result = self.model.predict_generator(
            self.build_generator(x, batch_size=batch_size, phase='test'),
            steps=steps, 
            **predict_args,
        )
                
        is_list = isinstance(probs_result, list)
        if not is_list:
            probs_result = [probs_result]
            for idx in range(len(probs_result)):
                probs = probs_result[idx]
                
                # check if binary classification
                if probs.shape[1] == 1:
                    # first column is probability of class 0 and second is of class 1
                    probs = np.hstack([1 - probs, probs])
        
                probs_result[idx] = probs
        
        return probs_result if is_list else probs_result[0]

    def score(self, x, y, **kwargs):
        """Returns the mean accuracy on the given test data and labels.
        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.
        # Returns
            score: float
                metric_score of predictions on `x` wrt. `y` (default: accuracy).
        # Raises
            ValueError: If the underlying model isn't configured to
                compute metric_score. You should pass `metrics=[metric_score]` to
                the `.compile()` method of the model.
        """
        y = np.asarray(y)
        evaluate_args = self.filter_sk_params(Sequential.evaluate_generator, kwargs)

        steps = evaluate_args.pop("steps", None)
        batch_size = self.sk_params.get("batch_size", 32)
        if steps is None:
            if x.shape[0] % batch_size != 0:
                batch_size, steps = x.shape[0], 1
            else:
                steps = int(x.shape[0] / batch_size)

        outputs = self.model.evaluate_generator(
            self.build_generator(x, y, batch_size=batch_size, phase='test'),
            steps=steps, 
            **evaluate_args,
        )
            
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == self.metric_score:
                return output
        raise ValueError('The model is not configured to compute %s. '
                         'You should pass `metrics=["%s"]` to '
                         'the `model.compile()` method.' % (self.metric_score, self.metric_score))
