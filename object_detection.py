from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D, GlobalMaxPool2D, Flatten, Dense, \
    BatchNormalization, Activation, Add, TimeDistributed
from keras.models import Model
import keras.backend as K
import tensorflow as tf


class RoIPool(Layer):
    """
    ROI pooling layer for 2D inputs.
        See Spatial Pyramid pooling in Deep Convolutional Networks for Visual Recognition, K. He, et al.

        # Arguments
            pool_size: size of pooling region to use, pool_size = 7 will result in a 7x7 region.
            num_rois: number of regions of interest to be used.

        # Input shape
            list of two 4D tensors [X_img, X_roi] with shape:
            X_img: `(1, rows, cols, channels)
            X_roi: `(1,num_rois,4)` list of rois, with ordering (x, y, w, h)

        # Output shape
            3D tensor with shape: `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoIPool, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        else:
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, inputs, mask=None):
        assert (len(inputs) == 2)
        img = inputs[0]
        rois = inputs[1]
        input_shape = K.shape(img)
        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            if self.dim_ordering is 'th':
                r = w / float(self.pool_size)
                c = h / float(self.pool_size)
                for i in range(self.pool_size):
                    for j in range(self.pool_size):
                        x1 = x + i * r
                        y1 = y + j * c
                        x2 = K.cast(x1 + r, 'int32')
                        y2 = K.cast(y1 + c, 'int32')
                        x1 = K.cast(x1, 'int32')
                        y1 = K.cast(y1, 'int32')
                        x2 = x1 + K.maximum(1, x2 - x1)
                        y2 = y1 + K.maximum(1, y2 - y1)
                        new_shape = [input_shape[0], input_shape[1], y2 - y1, x2 - x1]

                        rs = K.max(K.reshape(img[:, :, y1:y2, x1:x2], new_shape), axis=(2, 3))
                        outputs.append(rs)

            else:
                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')
                rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
                outputs.append(rs)

            outputs = K.concatenate(outputs, axis=0)
            outputs = K.reshape(outputs, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

            if self.dim_ordering == 'th':
                outputs = K.permute_dimensions(outputs, (0, 1, 4, 2, 3))
            else:
                outputs = K.permute_dimensions(outputs, (0, 1, 2, 3, 4))

            return outputs

    def get_config(self):
        config = {'pool_size': self.pool_size, 'num_rois': self.num_rois}
        base_config = super(RoIPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def RPN(inputs, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(inputs)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_reg = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_reg')(x)
    return [x_class, x_reg, inputs]


if __name__ == "__main__":
    pass
