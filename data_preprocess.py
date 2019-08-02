import tensorflow as tf 
import glob as gb
from cv2 import cv2  
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

train_path = "A:\\test_img_TFRecords\\img"
pattern = '\*.png'
Label_path = "A:\\test_img_TFRecords\\lab"
shuffle_data = True
save_path = 'A:\\test_img_TFRecords//.data.tfrecords'

class write_to_TFrecord:
    def __init__(self,FLAGS,image_type=None):
        if(image_type == None):
            self._type = '\*.png'
        else:
            self._type=image_type
        self._train_path =FLAGS.image_dir
        self._label_path =FLAGS.label_dir
        self._img_heigh =FLAGS.image_h
        self._img_width =FLAGS.image_w
        
        ##self.read_file(train_data_path,label_data_path,self._type)
        
    
    @property
    def Image_type(self):
        return self._type

    @Image_type.setter
    def Image_type(self,types):
        
        if (types =='png'):
            self._type = '\*.'+types
        elif (types=='bmp') :
            self._type = '\*.'+types
        elif (types == 'jpg'):
            self._type = '\*.'+types
        else:
            raise ValueError('Just identify:png,bmp,jpg')
                
    def read_file(self):
        img_list =[]
        label_list=[]
        for x in gb.glob(self._train_path+self._type):
            img_list.append(x)
        for w in gb.glob(self._label_path+self._type):
            label_list.append(w)
        if shuffle_data:
            c =list(zip(img_list,label_list))
            shuffle(c)
            img,label=zip(*c)
        return img,label

    def _Load_image_label(self,Img,Label):   
        img =cv2.imread(Img)###turn to one dimension

        img = cv2.resize(img, (self._img_width,self._img_heigh), interpolation=cv2.INTER_CUBIC)
        ##cv2.imshow("",img)
        ##cv2.waitKey(0)
        img =img.tostring()

        ##img =np.array2string(img)## revise
        label = cv2.imread(Label,cv2.IMREAD_GRAYSCALE)
        print(label.size)
        label = cv2.resize(label, (self._img_width,self._img_heigh), interpolation=cv2.INTER_CUBIC)

        label = label.tostring()
        ##label =np.array2string(label)
        return img,label 
    
    def transfer_to_TFrecord(self,img_list,label_list):

        ## feature dict depends on the data to change bytes ,strings,int .etc
        ## also  the "name" in feature group could be  changed and added other attributes that u wanna 

        writer = tf.python_io.TFRecordWriter(save_path)  
        for i in range(len(img_list)):## the image and label are read and tranfered one by one 
            img,label = self._Load_image_label(img_list[i],label_list[i])   
            
            feature = {'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label)])),
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img)]))}##define the key and value ,it likes dictinary 

            example = tf.train.Example(features=tf.train.Features(feature=feature))  ##build it 
            writer.write(example.SerializeToString()) ## serialize it 
        writer.close()
class Read_TFrecord:
    def __init__(self,FLAGS):
        self._tfrecord_path = FLAGS.TFrecord_dir
        self._img_heigh =FLAGS.image_h
        self._img_width =FLAGS.image_w
        self._Resize = FLAGS.img_resize
        self._batch_size =FLAGS.batch_size
    def Dataset_read(self):##first way 

        dataset = tf.data.TFRecordDataset(self._tfrecord_path)
        dataset =dataset.map(self._parse_function)##map func :the _parse_function as delivery parameter

        dataset =dataset.repeat().shuffle(self._batch_size)
        dataset=dataset.batch(self._batch_size)
    
        
        iterator = dataset.make_one_shot_iterator()
        next_element= iterator.get_next()
       
        # with tf.Session() as sess:
        #     init = tf.initialize_all_variables()
        #     sess.run(init)
        #     x,y =sess.run(next_element)
        #     print(tf.shape(x))
        #     plt.figure()
        #     plt.subplot(121)
        #     plt.imshow(x)
        #     y = np.reshape(y,[360,480])
        #     plt.subplot(122)
        #     plt.imshow(y)
        #     plt.show()
        #     print(x)
       
        return next_element
        
    def _parse_function(self,example_proto):
        features = {  
            'image': tf.FixedLenFeature([], tf.string),  
            'label': tf.FixedLenFeature([], tf.string)  
            }  
        parsed_features=tf.parse_single_example(example_proto,features)
        image = tf.decode_raw(parsed_features['image'], tf.uint8)
        
        image = tf.reshape(image,[self._img_heigh,self._img_width,3])
        image = tf.image.resize_images(image,[self._Resize,self._Resize],method=3)
        Label= tf.decode_raw(parsed_features['label'], tf.uint8)
        label = tf.reshape(Label,[self._img_heigh,self._img_width,1])
        label = tf.image.resize_images(label,[self._Resize,self._Resize],method=3)
        print(image)
        print(label)
        return image,label