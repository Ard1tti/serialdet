import numpy as np
import random as r
import time

import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging

import urllib
from StringIO import StringIO

import threading

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

invalid_font_list = ['KacstArt.ttf','KacstBook.ttf','KacstDecorative.ttf',
                    'KacstDigital.ttf','KacstFarsi.ttf','KacstLetter.ttf',
                    'KacstNaskh.ttf', 'KacstOffice.ttf','KacstPen.ttf',
                    'KacstPoster.ttf','KacstQurn.ttf','KacstScreen.ttf',
                    'KacstTitle.ttf','KacstTitleL.ttf','lklug.ttf',
                    'mry_KacstQurn.ttf','opens___.ttf']
ttfdir="/usr/share/fonts/"
ttflist=[]

for (path, dir, files) in os.walk(ttfdir):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.ttf' and not filename in invalid_font_list:
            ttflist.append("%s/%s" % (path, filename))

class Input(object):
    """ DataSet object """
    
    def __init__(self, files, labels, class_num, capacity=100, threads=16,
                 IMG_DIR=None, size=(224,224)):
        self.IMG_DIR = IMG_DIR
        self.files = files
        self.labels = labels

        self.IMG_N = len(labels)
        self.CLASS_NUM = class_num
        self.size=size
        
        self.capacity=capacity
        self._img_queue=[]
        self._lab_queue=[]
        self._threads=threads
        self._lock=threading.Lock()
        
        np.random.seed(int(time.time()))
    
    def batch(self, BATCH_NUM, coord=None):
        while True:
            assert BATCH_NUM < self.capacity
            if coord and coord.should_stop():
                break
            if len(self._lab_queue) > BATCH_NUM:
                with self._lock:
                    images = self._img_queue[0:BATCH_NUM]
                    labels = self._lab_queue[0:BATCH_NUM]
                    self._img_queue=self._img_queue[BATCH_NUM:]
                    self._lab_queue=self._lab_queue[BATCH_NUM:]
                return images, labels
    
    def train_batch(self, BATCH_NUM, coord=None):
        images, labels = self.batch(BATCH_NUM, coord)
        images=[self.augment(image) for image in images]
        return np.asarray(images,dtype=np.uint8), np.asarray(labels,dtype=np.int32)
    
    def eval_batch(self, BATCH_NUM=1, coord=None):
        images, labels = self.batch(BATCH_NUM, coord)
        images=[self.random_flip(self.random_resize(image)) for image in images]
        return np.asarray(images,dtype=np.uint8), np.asarray(labels,dtype=np.int32)
            
    def _run(self, coord=None):
        if coord:
            coord.register_thread(threading.current_thread())
        try:
            while True:
                if coord and coord.should_stop():
                    break
                
                if len(self._lab_queue)>self.capacity:
                    continue
                
                i = np.random.randint(0, self.IMG_N)
                file_name=self.files[i]
                label=self.labels[i]
            
                if self.IMG_DIR is None:
                    f = None
                    try:
                        f=urllib.urlopen(file_name)
                        img=Image.open(StringIO(f.read()))
                        image = np.array(img)
                        if image.shape[-1] is 3:
                            self._img_queue.append(image)
                            self._lab_queue.append(label)
                    except Exception as e:
                        continue
                else:
                    with open(self.IMG_DIR+file_name) as f:
                        img=Image.open(f)
                        image=np.array(img)
                        if image.shape[-1] is 3:
                            self._img_queue.append(image) 
                            self._lab_queue.append(label)
        except Exception as e:
            with self._lock:
                if coord:
                    logging.error(str(e))
                    coord.request_stop(e)
                else:
                    logging.error("Exception in QueueRunner: %s", str(e))
                    raise
                              
    def create_threads(self, coord=None, daemon=False, start=False):
        threads = [threading.Thread(target=self._run, args=(coord,)) 
                   for _ in xrange(self._threads)]
        for t in threads:
            if daemon: t.daemon = True
            if start: t.start()
        return threads
    
    def augment(self, image, short=[256,257], crop=[224,224]):
        image=self.random_resize(image)
        image=self.random_crop(image)
        image=self.random_flip(image)
        return image
    
    def augment_deprecated(self, image, short=[256,257], crop=[224,224]):
        img=Image.fromarray(image,'RGB')
        width, height = img.size
        
        target_short=np.random.randint(short[0],short[1])
        rate=target_short/float(min([width,height]))
        
        target_height=int(height*rate)
        target_width=int(width*rate)
        
        cut_height=np.random.randint(0,target_height-crop[0])
        cut_width=np.random.randint(0,target_width-crop[1])
        
        bool_flip=np.random.choice([True,False])
        
        img=img.resize((target_width,target_height),resample=Image.BILINEAR)
        img=img.crop((cut_width,cut_height,cut_width+crop[1],cut_height+crop[0]))
        if bool_flip:img=img.transpose(Image.FLIP_LEFT_RIGHT)
        
        res=np.array(img)
        #res=res-np.array([100,100,100])
        return res
    
    def random_flip(self, image):
        bool_flip=np.random.choice([True,False])
        img=Image.fromarray(image,"RGB")
        if bool_flip: img=img.transpose(Image.FLIP_LEFT_RIGHT)
        return np.array(img)
    
    def random_resize(self, image, short=[256,257]):
        img=Image.fromarray(image,'RGB')
        width, height = img.size
        
        target_short=np.random.randint(short[0],short[1])
        rate=target_short/float(min([width,height]))
        
        target_height=int(height*rate)
        target_width=int(width*rate)
        
        img=img.resize((target_width,target_height),resample=Image.BILINEAR)
        return np.array(img)
    
    def random_crop(self, image, crop=[224,224]):
        img=Image.fromarray(image,'RGB')
        width, height = img.size
        cut_height=np.random.randint(0,height-crop[0])
        cut_width=np.random.randint(0,width-crop[1])
        img=img.crop((cut_width,cut_height,cut_width+crop[1],cut_height+crop[0]))
        return np.array(img)
    
def label2vec(label, CLASS_NUM):
    v = np.zeros([CLASS_NUM])
    v[label]=1.0
    return v

def data_generator(n, size=(100,100), frange=(10,50)):
    data = [[],[]]
    stime=t.time()
    for i in range(n):
        if i % 100 == 0:
            ltime=t.time()-stime
            print("Data Generation : {0}/{1}, lapsed {2} sec, estimated {3} min ".format(i,n,ltime,ltime*n/(60*(i+1))))
        fsize=r.randrange(frange[0],frange[1])
        font = ImageFont.truetype(r.choice(ttflist),fsize)
        img = Image.new("L", size, 0)
        draw = ImageDraw.Draw(img)
        y_data = r.randrange(0, 10)
        draw.text((r.randrange(0,size[0]-fsize*4/3+1),r.randrange(0,size[1]-fsize*2/3+1)),
                  str(y_data), 255, font = font)
        draw = ImageDraw.Draw(img)
        x_data = np.reshape(img.getdata(),(1,)+size)
        data[0].append(x_data)
        data[1].append(vectorized_result(y_data))
    return zip(data[0], data[1])

       