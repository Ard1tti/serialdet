from imagenet_input_nq import Input
from coordinator import Coordinator
import pickle
import numpy as np
import models.ResNet18_imagenet as model
import tensorflow as tf
import os

BATCH_SIZE=32
EVAL_SIZE=1000
CLASS_NUM=1000
GPU_LIST=[0]
NUM_GPUS=len(GPU_LIST)
CKPT_DIR="../../ckpt/"+model.__name__+"/"
IMG_SIZE=[224,224]
IMG_DIR="../../data/images/"

def tower_accuracy(images, labels):
    logits = model.eval_once(images, CLASS_NUM)
    with tf.device('/cpu:0'):
        accu = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))

    return accu

def data_input(is_training=True):
    if is_training:
        f= open('../../data/urls/ILSVRC2012_val.txt','r')
        capacity=BATCH_SIZE*NUM_GPUS*3
        threads=BATCH_SIZE*NUM_GPUS

    else:
        f= open('../../data/urls/ILSVRC2012_val.txt','r')
        capacity=EVAL_SIZE*NUM_GPUS*3
        threads=EVAL_SIZE
        
    files, labels, class_list = pickle.load(f)
    f.close()
    
    files=np.asarray(files)
    labels=np.asarray(labels)
    
    data = Input(files,labels,class_num=CLASS_NUM,capacity=capacity,size=IMG_SIZE, threads=threads,
                IMG_DIR=IMG_DIR)
    return data

def test():
    # Test in multi GPU
    print('Testing '+model.__name__+' model')
    
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        val_data = data_input()
        
        val_images=[tf.placeholder(tf.float32, shape=[1,None,None,3])
                    for _ in xrange(NUM_GPUS)]
        val_labels=[tf.placeholder(tf.int32, shape=[1]) for _ in xrange(NUM_GPUS)]
        
        tower_accu = []
        
        for i in xrange(NUM_GPUS):
            with tf.device('/gpu:%d' % GPU_LIST[i]):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    accu = tower_accuracy(val_images[i],val_labels[i])
                    tf.get_variable_scope().reuse_variables()

                    tower_accu.append(accu)

        mean_accu = tf.reduce_mean(tower_accu)
       
        sess = tf.Session()
        coord = Coordinator()
        
        n_mean_a = 1
        highest = 0.
        mean_a = 0.
        
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Variables are restored from "+ CKPT_DIR)
        else:
            raise
            
        val_threads=val_data.create_threads(start=True, daemon=True, coord=coord)
        
        try:
            for batch_i in range(1):                       
                if (batch_i+1)%n_mean_a == 0:
                    for _ in xrange(EVAL_SIZE):
                        feed_dict={}
                        for i in xrange(NUM_GPUS):
                            img_batch, lab_batch = val_data.eval_batch(coord=coord)
                            feed_dict.update({val_images[i]: img_batch, val_labels[i]: lab_batch})
                            
                        mean_a = mean_a+sess.run(mean_accu, feed_dict = feed_dict)/EVAL_SIZE
                    if mean_a > highest:
                        highest = mean_a
                    print("test accuracy %g"%(mean_a))
                    mean_a=0.
            print("highest accuracy: %g"%(highest))
        except Exception as e:
            print(e)
        finally:
            coord.request_stop()
               
def main(argv=None):  # pylint: disable=unused-argument
    test()

if __name__ == '__main__':
    tf.app.run()
