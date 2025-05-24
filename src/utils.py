import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        p_t = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        loss = alpha * K.pow(1 - p_t, gamma) * K.sum(cross_entropy, axis=-1)
        return K.mean(loss)
    return focal_loss_fixed
