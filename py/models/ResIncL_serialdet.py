# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 23:58:29 2016

@author: bong
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:01:14 2016

@author: bong
"""

import tensorflow as tf
import numpy as np
import serialdet_layer as layer
    
def inference(images, keep_prop, is_training=False):
    
    conv1 = layer.conv2d_layer(images, [5, 5, 3, 64], name="conv1")
    norm1 = layer.bn_layer(conv1, is_training=is_training, activation_fn=tf.nn.relu, name="norm1")
        
    res1 = layer.resinc_layer(norm1, 64, 16,24,32,4,8,8, name="res1")

    conv2 = layer.conv2d_layer(res1, [3, 3, 64, 128], stride=2, name="conv2")
    norm2 = layer.bn_layer(conv2, is_training=is_training, activation_fn=tf.nn.relu, name="norm2")

    res2_1 = layer.resinc_layer(norm2, 128, 32,48,64,8,16,16, name="res2_1")
    res2 = layer.resinc_layer(res2_1, 128, 32,48,64,8,16,16, name="res2")

    conv3 = layer.conv2d_layer(res2, [3,3,128,256], stride=2, name="conv3")
    norm3 = layer.bn_layer(conv3, is_training=is_training, activation_fn=tf.nn.relu, name="norm3")
    
    res3_1 = layer.resinc_layer(norm3, 256, 64,96,128,16,32,32, name="res3_1")
    res3 = layer.resinc_layer(res3_1, 256, 64,96,128,16,32,32, name="res3")

    avg_pool = tf.nn.avg_pool(res3, ksize=[1,8,8,1], strides=[1,1,1,1],
                              padding='VALID', name="avg_pool")
    norm4 = layer.bn_layer(avg_pool, is_training=is_training, name="norm4")

    softmax_linear = layer.conv2d_layer(norm4, [1,1,256,CLASS_NUM], num="softmax")

    return tf.reshape(softmax_linear, [-1,CLASS_NUM])

def eval_once(images, CLASS_NUM, is_training=False):
    conv1 = layer.conv2d_layer(images, [5, 5, 3, 64], name="conv1")
    norm1 = layer.bn_layer(conv1, is_training=is_training, activation_fn=tf.nn.relu, name="norm1")
        
    res1 = layer.resinc_layer(norm1, 64, 16,24,32,4,8,8, name="res1")

    conv2 = layer.conv2d_layer(res1, [3, 3, 64, 128], stride=2, name="conv2")
    norm2 = layer.bn_layer(conv2, is_training=is_training, activation_fn=tf.nn.relu, name="norm2")

    res2_1 = layer.resinc_layer(norm2, 128, 32,48,64,8,16,16, name="res2_1")
    res2 = layer.resinc_layer(res2_1, 128, 32,48,64,8,16,16, name="res2")

    conv3 = layer.conv2d_layer(res2, [3,3,128,256], stride=2, name="conv3")
    norm3 = layer.bn_layer(conv3, is_training=is_training, activation_fn=tf.nn.relu, name="norm3")
    
    res3_1 = layer.resinc_layer(norm3, 256, 64,96,128,16,32,32, name="res3_1")
    res3 = layer.resinc_layer(res3_1, 256, 64,96,128,16,32,32, name="res3")

    avg_pool = tf.nn.avg_pool(res3, ksize=[1,8,8,1], strides=[1,1,1,1],
                              padding='VALID', name="avg_pool")
    norm4 = layer.bn_layer(avg_pool, is_training=is_training, name="norm4")

    softmax_linear = layer.conv2d_layer(norm4, [1,1,256,CLASS_NUM], num="softmax")
    softmax_linear = tf.reshape(softmax_linear, [-1,CLASS_NUM])
    return tf.reshape(tf.reduce_mean(softmax_linear, reduction_indices=[0]),[1,CLASS_NUM])

def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
def train(total_loss, lr):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op
