from imagenet_input_nq import Input
from coordinator import Coordinator
import pickle
import numpy as np
import models.ResIncL_serialdet as model
import tensorflow as tf
import os

BATCH_SIZE=100
EVAL_SIZE=100
CLASS_NUM=1000
GPU_LIST=[0]
NUM_GPUS=len(GPU_LIST)
CKPT_DIR="../../ckpt/"+model.__name__+"/"
IMG_SIZE=[224,224]
IMG_DIR="/data/ILSVRC2012/images/"

def tower_loss(images, labels, scope):
    logits = model.inference(images, CLASS_NUM)
    _ = model.loss(logits, labels)
    
    tf.get_variable_scope().reuse_variables()

    losses = tf.get_collection('losses', scope)
    
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

def tower_accuracy(images, labels):
    logits = model.eval_once(images, CLASS_NUM)
    with tf.device('/cpu:0'):
        accu = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))

    return accu

def average_grads(list_grads):
    grads = []
    for i in range(len(list_grads[0])):
        if list_grads[0][i][0] is None:
            grads.append((None, list_grads[0][i][1]))
        else:
            grads.append((tf.reduce_mean([list_grads[j][i][0]
                                          for j in range(len(list_grads))],[0]),
                         list_grads[0][i][1]))
    return grads

def data_input(is_training=True):
    if is_training:
        f= open('/data/ILSVRC2012/ILSVRC2012_val.txt','r')
        capacity=BATCH_SIZE*NUM_GPUS*3
        threads=BATCH_SIZE*NUM_GPUS

    else:
        f= open('/data/ILSVRC2012/ILSVRC2012_val.txt','r')
        capacity=EVAL_SIZE*NUM_GPUS*3
        threads=EVAL_SIZE
        
    files, labels, class_list = pickle.load(f)
    f.close()
    
    files=np.asarray(files)
    labels=np.asarray(labels)
    
    data = Input(files,labels,class_num=CLASS_NUM,capacity=capacity,size=IMG_SIZE, threads=threads,
                IMG_DIR=IMG_DIR)
    return data

def train():
    # Train in multi GPU
    print('Training '+model.__name__+' model')
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        train_data = data_input()
        val_data = data_input(False)
        
        images=[tf.placeholder(tf.float32, shape=[BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],3])
                for _ in xrange(NUM_GPUS)]
        labels=[tf.placeholder(tf.int32, shape=[BATCH_SIZE]) for _ in xrange(NUM_GPUS)]
        
        val_images=[tf.placeholder(tf.float32, shape=[1,None,None,3])
                    for _ in xrange(NUM_GPUS)]
        val_labels=[tf.placeholder(tf.int32, shape=[1]) for _ in xrange(NUM_GPUS)]
        
        tower_grads = []
        tower_losses = []
        tower_accu = []
        
        opt = tf.train.MomentumOptimizer(0.01, 0.9)
        
        for i in xrange(NUM_GPUS):
            with tf.device('/gpu:%d' % GPU_LIST[i]):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    loss = tower_loss(images[i],labels[i], scope)
                    tf.get_variable_scope().reuse_variables()
                    accu = tower_accuracy(val_images[i],val_labels[i])
                    
                    grads = opt.compute_gradients(loss)
                    tower_accu.append(accu)
                    tower_losses.append(loss)
                    tower_grads.append(grads)
                    
        grads = average_grads(tower_grads)
        mean_loss = tf.reduce_mean(tower_losses)
        mean_accu = tf.reduce_mean(tower_accu)
       
        train_op = opt.apply_gradients(grads)
    
        sess = tf.Session()
        coord = Coordinator()
        
        n_mean = 5
        n_mean_a = 5
        
        mean = 0.
        mean_a=0.
        highest = 0.
        
        saver = tf.train.Saver(tf.all_variables())
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)
        
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Variables are restored from "+ CKPT_DIR)
        else:
            sess.run(tf.initialize_all_variables())
            print("Variables are initialized")
            
        threads=train_data.create_threads(start=True, daemon=True, coord=coord)
        val_threads=val_data.create_threads(start=True, daemon=True, coord=coord)
        
        try:
            for batch_i in range(20000):
                feed_dict={}
                for i in xrange(NUM_GPUS):
                    while True:
                        img_batch, lab_batch = train_data.train_batch(BATCH_SIZE, coord)
                        if img_batch.shape[0]==BATCH_SIZE and lab_batch.shape[0]==BATCH_SIZE:
                            feed_dict.update({images[i]: img_batch, labels[i]: lab_batch})
                            break
                            
                _, cross_entropy = sess.run([train_op, mean_loss],
                                                       feed_dict = feed_dict)
                mean = mean + cross_entropy/n_mean
                
                if (batch_i+1)%n_mean == 0:
                    print("step %d, cross_entropy %g"%(batch_i+1, mean))
                    mean=0.
                        
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
                
                if (batch_i+1)%500 == 0:
                    checkpoint_path = os.path.join(CKPT_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path)
                    print("Model saved in "+CKPT_DIR)
            print("highest accuracy: %g"%(highest))
        except Exception as e:
            print(e)
        finally:
            coord.request_stop()
            checkpoint_path = os.path.join(CKPT_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_path)
            print("Model saved in "+CKPT_DIR)
               
def main(argv=None):  # pylint: disable=unused-argument
    train()

if __name__ == '__main__':
    tf.app.run()
