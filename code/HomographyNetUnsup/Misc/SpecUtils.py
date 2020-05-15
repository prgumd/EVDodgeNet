#!/usr/bin/env python

# Code adapted from https://github.com/daniilidis-group/polar-transformer-networks
import tensorflow as tf
import numpy as np

def polar_transformer(U, theta, out_size, name='polar_transformer',
                      log=True, radius_factor=0.5):
    """Polar Transformer Layer

    Based on https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py.
    _repeat(), _interpolate() are exactly the same;
    the polar transform implementation is in _transform()

    Args:
        U, theta, out_size, name: same as spatial_transformer.py
        log (bool): log-polar if True; else linear polar
        radius_factor (float): fraction of width that will be the maximum radius
    """
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = out_size[0]# np.shape(im)[0]
            height = 128# out_size[1] # np.shape(im)[1]
            width = 128# out_size[2] # np.shape(im)[2]
            channels = 3# out_size[3] # np.shape(im)[3]
    

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = 128 # out_size[1] # 0
            out_width = 128# out_size[2] # 1
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(128 - 1, 'int32')
            max_x = tf.cast(128 - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat])

            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = out_size[0] # np.shape(input_dim)[0]
            num_channels = 3 # out_size[3] # np.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2))
            theta = tf.cast(theta, 'float32')

            out_height = 128 # out_size[1] # 0
            out_width = 128 # out_size[2] # 1
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 2, -1]))

            ## here we do the polar/log-polar transform

            W = tf.cast(out_size[1], 'float32') # tf.cast(np.shape(input_dim)[1], 'float32')
            maxR = W*radius_factor

            # if radius is from 1 to W/2; log R is from 0 to log(W/2)
            # we map the -1 to +1 grid to log R
            # then remap to 0 to 1
            
            # get radius in pix
            if log:
                # min=1, max=maxR
                r_s = tf.exp((grid[:, 0, :] + 1)/2*tf.log(maxR))
            else:
                # min=1, max=maxR
                r_s = 1 + (grid[:, 0, :] + 1)/2*(maxR-1)
            # convert it to [0, 2maxR/W]
            r_s = (r_s - 1) / (maxR - 1) * 2 * maxR / W
            # y is from -1 to 1; theta is from 0 to 2pi
            t_s = (grid[:, 1, :] + 1)*np.pi

            # use + theta[:, 0] to deal with origin
            x_s = r_s*tf.cos(t_s) + theta[:, 0, np.newaxis]
            y_s = r_s*tf.sin(t_s) + theta[:, 1, np.newaxis]

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)
            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output
