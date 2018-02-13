import keras
from keras.initializers import Constant
from keras.layers import Layer
from keras.models import Model
from keras import backend as K

import numpy as np


class AutoMultiLossLayer(Layer):

    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(AutoMultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # initialise log_vars
        # keep the tensor used in the automatic weighting multi-loss
        self.log_vars = [self.add_weight(name='log_var' + str(idx), shape=(1,),
                                         initializer=Constant(2.),
                                         trainable=True)
                         for idx in range(self.nb_outputs)]

        super(AutoMultiLossLayer, self).build(input_shape)

    def call(self, inputs):
        # don't use inputs, returns internal weights
        return K.concatenate(self.log_vars, -1)

    def get_auto_weighted_loss(self, index, loss_func):
        assert 0 <= index < self.nb_outputs
        # compute log(sigma^2)
        log_var = self.log_vars[index][0]

        # weighted loss for indexed weight
        def _loss(y_true, y_pred):
            # TODO: check if the formula is correct
            # Expected formula: (1 / (2 * sigma^2)) * loss_func(y_true, y_pred) + log(sigma^2)
            return K.exp(-log_var) * loss_func(y_true, y_pred) + log_var

        return _loss

    def get_sigma(self):
        # sigma = sqrt(exp(log_var))
        return [np.exp(K.get_value(log_var[0])) ** 0.5 for log_var in
                self.log_vars]


class AutoMultiLossWrapper(object):

    def __init__(self, model):
        self.original_model = model
        self.model = None
        self._auto_weighted_loss = False

    def compile(self, optimizer, loss, loss_weights=None, **kwargs):
        self._auto_weighted_loss = False

        self.model = self.original_model
        # compute auto multi-loss only when loss is specified and
        # there are atleast two outputs
        if loss_weights == 'auto' and loss and len(self.model.outputs) > 1:
            n_losses = len(self.model.outputs)
            dummy_layer = AutoMultiLossLayer(n_losses,
                                             name="auto_loss_w")(
                self.model.outputs)
            # add a dummy output to the graph to make the weights trainable
            self.model = Model(self.model.inputs,
                               self.model.outputs + [dummy_layer])

            loss = self._get_losses(loss)

            self._auto_weighted_loss = True

        # fix parameter for keras api
        # by setting loss_weights to None, keras treats all losses equally
        if loss_weights == 'auto':
            loss_weights = None

        self.model.compile(optimizer, loss, loss_weights=loss_weights, **kwargs)

    def _get_losses(self, loss):
        loss_factory = self.model.get_layer(
            "auto_loss_w").get_auto_weighted_loss
        # deals with losses in the same way that keras model.compile method
        # allows compatibility
        if isinstance(loss, dict):
            loss_keys = sorted(loss.keys())

            new_loss = {
            name: loss_factory(loss_idx, keras.losses.get(loss.get(name)))
            for loss_idx, name in enumerate(loss_keys)}
            # keras doesn't consider an output when its loss is set to None
            # skip dummy output
            new_loss['auto_loss_w'] = None

        elif isinstance(loss, list):
            new_loss = [loss_factory(loss_idx, keras.losses.get(name))
                        for loss_idx, name in enumerate(loss)]
            # skip dummy output
            new_loss.append(None)

        else:
            loss_func = keras.losses.get(loss)
            new_loss = [loss_factory(loss_idx, loss_func)
                        for loss_idx in range(len(self.original_model.outputs))]
            # skip dummy output
            new_loss.append(None)

        return new_loss

    def get_sigma(self):
        if self._auto_weighted_loss:
            return self.model.get_layer('auto_loss_w').get_sigma()
        else:
            return None
