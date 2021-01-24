from keras.layers import Layer
from keras import backend as K


class SampleLayer(Layer):
    def __init__(self):
        super(SampleLayer, self).__init__()

    def build(self, input_shape):
        # save the shape for distribution sampling
        super(SampleLayer, self).build(input_shape) # needed for layers

    def call(self, x, training=None):
        mean = x[0]
        logvar = x[1]

        # trick to allow setting batch at train/eval time
        if mean.shape[0].value == None or  logvar.shape[0].value == None:
            return mean + 0*logvar # Keras needs the *0 so the gradinent is not None
        '''
        # kl divergence:
        latent_loss = -0.5 * (1 + logvar
                            - K.square(mean) 
                            - K.exp(logvar))
        latent_loss = K.sum(latent_loss, axis=-1) # sum over latent dimension
        latent_loss = K.mean(latent_loss, axis=0) # avg over batch

        # use beta to force less usage of vector space:
        latent_loss = latent_loss
        self.add_loss(latent_loss, x)
        '''
        def reparameterization_trick():
            epsilon = K.random_normal(shape=logvar.shape,
                              mean=0., stddev=1.)
            stddev = K.exp(logvar*0.5)
            return mean + stddev * epsilon

        return K.in_train_phase(reparameterization_trick, mean + 0*logvar, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape[0]