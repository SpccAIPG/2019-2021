from keras import initializers, regularizers, constraints, optimizers
from keras import backend as K
from keras.layers import Layer, InputSpec
from numpy.random import randint
import numpy as np
import tensorflow as tf

class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center: 
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def gram_matrix(x):
    norm_by_channels = False
    '''
    Returns the Gram matrix of the tensor x.
    '''
    if K.ndim(x) == 3:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        shape = K.shape(x)
        C, H, W = shape[0], shape[1], shape[2]
        gram = K.dot(features, K.transpose(features))
    elif K.ndim(x) == 4:
        # Swap from (H, W, C) to (B, C, H, W)
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        # Reshape as a batch of 2D matrices with vectorized channels
        features = K.reshape(x, K.stack([B, C, H*W]))
        # This is a batch of Gram matrices (B, C, C).
        gram = K.batch_dot(features, features, axes=2)
    else:
        raise ValueError('The input tensor should be either a 3d (H, W, C) or 4d (B, H, W, C) tensor.')
    # Normalize the Gram matrix
    if norm_by_channels:
        denominator = C * H * W # Normalization from Johnson
    else:
        denominator = H * W # Normalization from Google
    gram = gram /  K.cast(denominator, x.dtype)

    return gram

    
def create_comparison_img_content(imgs, n_words, font_no): # meaning the comparison img used in training the generating content vector from the pre-trained style vector
    img_ls = []
    
    for font in range(font_no):
        font_ls = []
        rand_ls = np.arange(n_words)
        np.random.shuffle(rand_ls)
        for word in rand_ls:
            font_ls.append(imgs[font][word])
        
        img_ls.append(font_ls)

    return np.array(img_ls)

def create_comparison_img_style(imgs, comp_imgs, n_words, font_no): # meaning the comparison img used in training the generating style vector from the pre-trained content vector
    img_ls = []
    comp_img_ls = []
    rand_ls = np.arange(font_no)
    np.random.shuffle(rand_ls)

    for font in rand_ls:
        font_ls = []
        comp_font_ls = []
        for word in range(n_words):
            font_ls.append(imgs[font][word])
            comp_font_ls.append(comp_imgs[font][word])
        
        img_ls.append(font_ls)
        comp_img_ls.append(comp_font_ls)

    return np.array(img_ls), np.array(comp_img_ls)


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.get_output_at(layer))
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad