import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm

def _variable_on_cpu(shape, init, name, dtype=tf.float32, trainable=True):
    with tf.device("/cpu:0"):
        return tf.get_variable(name, shape, dtype=dtype,
                               initializer=init, trainable=trainable)

#variables for convolutional network
def weight_variable(shape, stdev=0.05, wd=0.0):
    init = tf.truncated_normal_initializer(stddev=stdev, dtype=tf.float32)
    var = _variable_on_cpu(shape=shape, init=init, name='weights')
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def bias_variable(shape, const=0.1):
    init = tf.constant_initializer(const)
    return _variable_on_cpu(shape=shape, init=init, name='biases')

#layers
def conv2d_layer(x, shape, stride=1, name='conv2d', activation_fn=None):
    with tf.variable_scope(name) as scope:
        kernel = weight_variable(shape=shape)
        biases = bias_variable(shape=shape[3])
        conv = tf.nn.conv2d(x, kernel, [1,stride,stride,1], padding='SAME')
        outputs = tf.nn.bias_add(conv,biases)
        if activation_fn:
            outputs=activation_fn(outputs)
        return outputs

def max_pool_layer(x, stride=2, name='max_pool'):
    with tf.variable_scope(name) as scope:
        outputs=tf.nn.max_pool(x, name=name, ksize=[1,3,3,1],
                               strides=[1,stride,stride,1],padding='SAME')
        return outputs

def inception_layer(x, width, ker1, red3, ker3, red5, ker5, pool, name='inception'):
    with tf.variable_scope(name) as scope:
        conv1 = conv2d_layer(x, [1, 1, width, ker1], name='conv1', 
                            activation_fn=tf.nn.relu)
    
        red_3 = conv2d_layer(x, [1, 1, width, red3], name='red3')
        conv3 = conv2d_layer(red_3, [3, 3, red3, ker3], name='conv3',
                            activation_fn=tf.nn.relu)
        
        red_5 = conv2d_layer(x, [1, 1, width, red5], name='red5')
        conv5 = conv2d_layer(red_5, [5, 5, red5, ker5], name='conv5',
                            activation_fn=tf.nn.relu)
    
        pooled = max_pool_layer(x, stride=1, name='pooled')
        pool_conv = conv2d_layer(pooled, [1, 1, width, pool], name='pool_conv',
                                activation_fn=tf.nn.relu)
        
        outputs= tf.concat(3, [conv1, conv3, conv5, pool_conv], name=name)
        return outputs

def residual_layer(x, shape, name='residual', activation_fn=tf.nn.relu):
    with tf.variable_scope(name) as scope:
        assert shape[2]==shape[3]    
        conv1 = conv2d_layer(x, shape, name='conv1', activation_fn=tf.nn.relu)
        conv2 = conv2d_layer(conv1, shape, name='conv2')
        outputs = tf.add(x,conv2, name=scope.name)
        if activation_fn:
            outputs = activation_fn(outputs)
        return outputs

def resinc_layer(x, width, ker1, red3, ker3, red5, ker5, pool, ker=3, 
                 name='residual', activation_fn=tf.nn.relu):
    with tf.variable_scope(name) as scope:
        inc_out = ker1+ker3+ker5+pool
        shape=[ker,ker,inc_out,width]
    
        inception = inception_layer(x, width, ker1, red3, ker3, red5, ker5, pool,
                                    name='inception')
        conv = conv2d_layer(inception, shape, name='conv')
        outputs = tf.add(x,conv, name=scope.name)
        if activation_fn:
            outputs=activation_fn(outputs) 
        return outputs

def bn_layer(inputs, is_training=True, trainable=True, activation_fn=None,
             decay=0.999, center=True, scale=False, epsilon=0.001, name="BatchNorm"):
    with tf.variable_scope(name) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        axis = list(range(inputs_rank - 1))
        params_shape = inputs_shape[-1:]
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                    inputs.name, params_shape))
    
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center: beta = _variable_on_cpu(shape=params_shape,
                                           init=tf.zeros_initializer, name='beta')
        if scale: gamma = _variable_on_cpu(shape=params_shape,
                                           init=tf.ones_initializer, name='gamma')
        
        # Create moving_mean and moving_variance variables and add them to the
        # appropiate collections.
        moving_mean = _variable_on_cpu(shape=params_shape,
                                       init=tf.zeros_initializer, trainable=False, 
                                       name='moving_mean')
        moving_var = _variable_on_cpu(shape=params_shape,
                                      init=tf.ones_initializer, trainable=False,
                                      name='moving_var')

        # If `is_training` doesn't have a constant value, because it is a `Tensor`,
        # a `Variable` or `Placeholder` then is_training_value will be None and
        # `needs_moments` will be true.
        if is_training:
            # Calculate the moments based on the individual batch.
            # Use a copy of moving_mean as a shift to compute more reliable moments.
            shift = tf.add(moving_mean, 0)
            mean, variance = tf.nn.moments(inputs, axis, shift=shift)
            with tf.device("/cpu:0"):
                update_moving_mean = moving_mean.assign(tf.div(
                        tf.add(tf.mul(decay,moving_mean),mean),1+decay))
                update_moving_var= moving_var.assign(tf.div(
                        tf.add(tf.mul(decay,moving_var),variance),1+decay))
            with tf.control_dependencies([update_moving_mean,
                                          update_moving_var]):
                mean, variance = tf.identity(mean), tf.identity(variance)
        else:
            mean, variance = moving_mean, moving_var
    
        # Compute batch_normalization.
        outputs = tf.nn.batch_normalization(
                inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs_shape)
        if activation_fn:
            outputs = activation_fn(outputs)
        return outputs