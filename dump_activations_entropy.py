
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import skimage.io as io
import vgg_preprocessing as preprocess
import numpy as np
from tqdm import tqdm

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# get_ipython().magic(u'matplotlib inline')

# Hyper-parameters

# DATA
IMAGE_HEIGHT = 224                                             # default image size taken in vgg_16
IMAGE_WIDTH = 224
# num_data_samples = 52*50                                       # 1.2 Million train, 50000 validation
num_data_samples = 66516                                     # 1.2 Million train, 50000 validation
data_dir = "/deepack/model_tailoring/tf_records/tailored_data_new_labels/"
# checkpoint = '/deepack/model_tailoring/slim_models/vgg_16.ckpt'
checkpoint = '/deepack/model_tailoring/final_models/model_finetune_52C_87_77/model_finetune_52/model.ckpt'
entropy_dumps = '/deepack/model_tailoring/entropy_dumps/'
dataset_split_name = "train"  

# TRAIN
batch_size = 46
num_threads = 5                                                # For tf.train.batch
learning_rate = 0.0
is_training = False
keep_prob6 = 1
keep_prob7 = 1
num_epochs = 1
num_batches = int(num_data_samples/batch_size)
filters_11 = 497
filters_12 = 467
filters_13 = 241
num_classes = 52

def weight_variable(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(shape,name):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
def l1_norm(x):
    return tf.reduce_sum(tf.abs(x), axis=[0,1,2])
def fetch_images(serialized_example, IMAGE_HEIGHT=224, IMAGE_WIDTH=224):
    features = tf.parse_single_example(
          serialized_example, features = {
          'image/encoded': tf.FixedLenFeature(
              (), tf.string, default_value=''),
          'image/format': tf.FixedLenFeature(
              (), tf.string, default_value='jpeg'),
          'image/class/label': tf.FixedLenFeature(
              [], dtype=tf.int64, default_value=-1),
          'image/class/text': tf.FixedLenFeature(
              [], dtype=tf.string, default_value=''),
          'image/height' : tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/width' :  tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/channels' :  tf.FixedLenFeature(
            [],dtype=tf.int64),
          'image/object/class/label': tf.VarLenFeature(
              dtype=tf.int64),
      })

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    label = tf.cast(features['image/class/label'],tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    num_channels = tf.cast(features['image/channels'], tf.int32)
    image_shape = tf.stack([height,width,num_channels])
    image = tf.cast(image, tf.float32)
    preprocessed_image = preprocess.preprocess_image(image,IMAGE_HEIGHT,IMAGE_WIDTH,is_training=is_training)
    images, labels = tf.train.batch( [preprocessed_image, label],
                                                     batch_size=batch_size,
                                                     capacity=batch_size+num_threads*batch_size,
                                                     num_threads=num_threads)
    return images,labels

def get_data_files():
    tfrecords_filename = []
    if dataset_split_name == 'train':
        length = 128
        data = "-of-00128"
    if dataset_split_name == 'validation':
        length = 8
        data = "-of-00008"

    for k in range(length): # Train data tfrecords
        j = "-0%04d"%k        # pad with 0's
        tfrecords_filename.append(data_dir+dataset_split_name+j+data)
    return tfrecords_filename

def network(images,is_training,keep_prob6,keep_prob7,
            fc_conv_padding='VALID',classes=1000):
    with tf.variable_scope('vgg_16'):
        with tf.variable_scope('conv1'):
            W_conv1_1 = weight_variable([3,3,3,64], name="conv1_1/weights")
            b_conv1_1 = bias_variable([64], name="conv1_1/biases")
            W_conv1_2 = weight_variable([3,3,64,64],name="conv1_2/weights")
            b_conv1_2 = bias_variable([64],name="conv1_2/biases")

        with tf.variable_scope('conv2'):
            W_conv2_1 = weight_variable([3,3,64,128],name="conv2_1/weights")
            b_conv2_1 = bias_variable([128],name="conv2_1/biases")
            W_conv2_2 = weight_variable([3,3,128,128],name="conv2_2/weights")
            b_conv2_2 = bias_variable([128],name="conv2_2/biases")

        with tf.variable_scope('conv3'):
            W_conv3_1 = weight_variable([3,3,128,256],name="conv3_1/weights")
            b_conv3_1 = bias_variable([256],name="conv3_1/biases")
            W_conv3_2 = weight_variable([3,3,256,256],name="conv3_2/weights")
            b_conv3_2 = bias_variable([256],name="conv3_2/biases")
            W_conv3_3 = weight_variable([3,3,256,256],name="conv3_3/weights")
            b_conv3_3 = bias_variable([256],name="conv3_3/biases")

        with tf.variable_scope('conv4'):
            W_conv4_1 = weight_variable([3,3,256,512],name="conv4_1/weights")
            b_conv4_1 = bias_variable([512],name="conv4_1/biases")
            W_conv4_2 = weight_variable([3,3,512,512],name="conv4_2/weights")
            b_conv4_2 = bias_variable([512],name="conv4_2/biases")
            W_conv4_3 = weight_variable([3,3,512,512],name="conv4_3/weights")
            b_conv4_3 = bias_variable([512],name="conv4_3/biases")

        with tf.variable_scope('conv5'):
            W_conv5_1 = weight_variable([3,3,512,512],name="conv5_1/weights")
            b_conv5_1 = bias_variable([512],name="conv5_1/biases")
            W_conv5_2 = weight_variable([3,3,512,512],name="conv5_2/weights")
            b_conv5_2 = bias_variable([512],name="conv5_2/biases")
            W_conv5_3 = weight_variable([3,3,512,512],name="conv5_3/weights")
            b_conv5_3 = bias_variable([512],name="conv5_3/biases")


        W_fc6 = weight_variable([7,7,512,4096],name="fc6/weights")
        b_fc6 = weight_variable([4096],name="fc6/biases")
        W_fc7 = weight_variable([1,1,4096,4096],name="fc7/weights")
        b_fc7 = bias_variable([4096],name="fc7/biases")
        W_fc8 = weight_variable([1,1,4096,num_classes],name="fc8/weights")
        b_fc8 = bias_variable([num_classes],name="fc8/biases")

    #Inference
    a_conv1_1 = tf.nn.bias_add(conv2d(images, W_conv1_1),b_conv1_1)
    h_conv1_1 = tf.nn.relu(a_conv1_1)
    a_conv1_2 = tf.nn.bias_add(conv2d(h_conv1_1, W_conv1_2),b_conv1_2)
    h_pool1   = max_pool_2x2(tf.nn.relu(a_conv1_2))

    a_conv2_1 = tf.nn.bias_add(conv2d(h_pool1, W_conv2_1),b_conv2_1)
    h_conv2_1 = tf.nn.relu(a_conv2_1)
    a_conv2_2 = tf.nn.bias_add(conv2d(h_conv2_1, W_conv2_2),b_conv2_2)
    h_pool2   = max_pool_2x2(tf.nn.relu(a_conv2_2))

    a_conv3_1 = tf.nn.bias_add(conv2d(h_pool2, W_conv3_1),b_conv3_1)
    h_conv3_1 = tf.nn.relu(a_conv3_1)
    a_conv3_2 = tf.nn.bias_add(conv2d(h_conv3_1, W_conv3_2),b_conv3_2)
    h_conv3_2 = tf.nn.relu(a_conv3_2)
    a_conv3_3 = tf.nn.bias_add(conv2d(h_conv3_2, W_conv3_3),b_conv3_3)
    h_pool3   = max_pool_2x2(tf.nn.relu(a_conv3_3))

    a_conv4_1 = tf.nn.bias_add(conv2d(h_pool3, W_conv4_1),b_conv4_1)
    h_conv4_1 = tf.nn.relu(a_conv4_1)
    a_conv4_2 = tf.nn.bias_add(conv2d(h_conv4_1, W_conv4_2),b_conv4_2)
    h_conv4_2 = tf.nn.relu(a_conv4_2)
    a_conv4_3 = tf.nn.bias_add(conv2d(h_conv4_2, W_conv4_3),b_conv4_3)
    h_pool4   = max_pool_2x2(tf.nn.relu(a_conv4_3))

    a_conv5_1 = tf.nn.bias_add(conv2d(h_pool4, W_conv5_1),b_conv5_1)
    h_conv5_1 = tf.nn.relu(a_conv5_1)
    a_conv5_2 = tf.nn.bias_add(conv2d(h_conv5_1, W_conv5_2),b_conv5_2)
    h_conv5_2 = tf.nn.relu(a_conv5_2)
    a_conv5_3 = tf.nn.bias_add(conv2d(h_conv5_2, W_conv5_3),b_conv5_3)
    h_pool5   = max_pool_2x2(tf.nn.relu(a_conv5_3))

    # In place of FC, use conv2d with VALID padding

    a_fc6 = tf.nn.bias_add(tf.nn.conv2d(h_pool5, W_fc6, strides = [1,1,1,1],padding=fc_conv_padding),b_fc6)
    h_fc6 = tf.nn.relu(a_fc6)
    d_fc6 = tf.nn.dropout(h_fc6,keep_prob=keep_prob6)

    a_fc7 = tf.nn.bias_add(conv2d(d_fc6, W_fc7),b_fc7)
    h_fc7 = tf.nn.relu(a_fc7)
    d_fc7 = tf.nn.dropout(h_fc7,keep_prob=keep_prob7)

    a_fc8  = tf.nn.bias_add(conv2d(d_fc7, W_fc8),b_fc8)
    logits = tf.squeeze(a_fc8,[1,2])

    s1 = tf.nn.relu(a_conv1_1)
    s2 = tf.nn.relu(a_conv1_2)
    s3 = tf.nn.relu(a_conv2_1)
    s4 = tf.nn.relu(a_conv2_2)
    s5 = tf.nn.relu(a_conv3_1)
    s6 = tf.nn.relu(a_conv3_2)
    s7 = tf.nn.relu(a_conv3_3)
    s8 = tf.nn.relu(a_conv4_1)
    s9 = tf.nn.relu(a_conv4_2)
    s10 = tf.nn.relu(a_conv4_3)
    s11 = tf.nn.relu(a_conv5_1)
    s12 = tf.nn.relu(a_conv5_2)
    s13 = tf.nn.relu(a_conv5_3) 

    return logits, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13
    # return logits, h_conv1_1, h_conv1_2, h_conv2_1, h_conv2_2, h_conv3_1, h_conv3_2, h_conv3_3, h_conv4_1, h_conv4_2, h_conv4_3, h_conv5_1, h_conv5_2, h_conv5_3 

def configure_lr(global_step,learning_rate):
    decay_steps = int(num_data_samples / batch_size *
                    num_epochs_per_decay)
    return tf.train.exponential_decay(learning_rate,global_step,
                    decay_steps,learning_rate_decay_factor,
                    staircase=True,name='exponential_decay_learning_rate')

tf.logging.set_verbosity(tf.logging.INFO)

# #placeholders
# reinforce_baseline  = tf.placeholder(tf.float32)
# Input pipeline
list_filenames = get_data_files()
filename_queue = tf.train.string_input_producer(
     list_filenames, capacity=batch_size, shuffle=False )
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
[images,labels] = fetch_images(serialized_example)
labels -= 1

# Inference
logits, a1_1, a1_2, a2_1, a2_2, a3_1, a3_2, a3_3, a4_1, a4_2, a4_3, a5_1, a5_2, a5_3 = network(images,is_training,keep_prob6,keep_prob7)

labels = tf.cast(labels,tf.int64)
lbl_one_hot = tf.one_hot(labels, num_classes, 1.0, 0.0)
labels = tf.squeeze(labels)
prediction = tf.argmax(logits,1)
correct_prediction = tf.reduce_sum(tf.cast(tf.equal(prediction,labels),tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# graph ends ! you can create session now!
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

import cPickle as cp
temp_var = True
with tf.Session(config=config) as sess:
    sess.run(init_op)
    # train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
    list_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
    saver = tf.train.Saver(list_vars)
    saver.restore(sess, checkpoint)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)
    acc = 0
    for itr, i in enumerate(range(num_batches)):
        a, c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c3_3, c4_1, c4_2, c4_3, c5_1, c5_2, c5_3, out_labels, out_predictions = sess.run([correct_prediction, a1_1, a1_2, a2_1, a2_2, a3_1, a3_2, a3_3, a4_1, a4_2, a4_3, a5_1, a5_2, a5_3, labels, prediction])

        temp_list = [c1_1, c1_2, c2_1, c2_2, c3_1, c3_2, c3_3, c4_1, c4_2, c4_3, c5_1, c5_2, c5_3]

        dump_list = []
        for item in temp_list:
            item = np.sum(item, axis=(1,2))        
            dump_list.append(item)

        if itr==0:
            dump_layers = dump_list
            dump_labels = out_labels
            dump_predictions = out_predictions
        else:      
            for idx, m in enumerate(dump_list):
                dump_layers[idx] = np.vstack((dump_layers[idx], dump_list[idx]))
            dump_labels = np.hstack((dump_labels, out_labels))
            dump_predictions = np.hstack((dump_predictions, out_predictions))

        print('Done with batch ' + str(itr))
        acc += a
    print(acc/num_data_samples)
    coord.request_stop()
    coord.join(threads)

    with open(entropy_dumps+'52C_finetune'+'.save', 'wb') as pf:
        cp.dump([dump_layers, dump_labels, dump_predictions], pf, cp.HIGHEST_PROTOCOL)
