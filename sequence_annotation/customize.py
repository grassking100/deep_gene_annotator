from keras.layers import Layer

class Mask(Layer):
    """Get mask of the previous layer"""
    def __init__(self, **kwargs):
        self.supports_masking = True
        super().__init__(**kwargs)

    def compute_mask(self, input_, input_mask=None):
        return input_mask

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
        return mask

def symmetric_sigmoid(x):
    return K.hard_sigmoid(((x-0.5)*5))

def sgn(x):
    return tf.cast(x >= 0,dtype='float32')*2-1

def noised_symmetric_sigmoid(x,alpha=1,c=1,p=1):
    train_phase = tf.cast(keras.backend.learning_phase(),dtype='float32')
    raw_result = symmetric_sigmoid(x)
    delta = raw_result - x
    d = -sgn(x) * sgn(1- alpha)
    sigma = c * (K.sigmoid(p * delta) - 0.5)**2
    iid_noise = keras.backend.random_normal(shape=(1,tf.shape(x)[-1]))
    noised_result = alpha*raw_result + (1-alpha)*x + d*sigma*iid_noise
    return train_phase*(noised_result)+(1-train_phase)*raw_result

def noised_relu(x,alpha=1,c=1,p=1):
    train_phase = tf.cast(keras.backend.learning_phase(),dtype='float32')
    raw_result = Activation('relu')(x)
    delta = raw_result - x
    d = -sgn(x) * sgn(1- alpha)
    sigma = c * (K.sigmoid(p * delta) - 0.5)**2
    iid_noise = keras.backend.random_normal(shape=(1,tf.shape(x)[-1]))
    noised_result = alpha*raw_result + (1-alpha)*x + d*sigma*iid_noise
    return train_phase*(noised_result)+(1-train_phase)*raw_result
class MaskedConvolution1D(Conv1D):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.supports_masking = True