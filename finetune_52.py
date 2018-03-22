
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import skimage.io as io
import vgg_preprocessing as preprocess
import numpy as np
from tqdm import tqdm
from IPython import embed
import datetime



IMAGE_HEIGHT = 224   
IMAGE_WIDTH = 224
num_data_samples = 66516

data_dir = "/deepack/model_tailoring/tf_records/tailored_data_new_labels/"
save_dir = "/deepack/model_tailoring/saved_models/train"
checkpoint = '/deepack/model_tailoring/slim_models/vgg_16.ckpt'
summary_location = '/deepack/model_tailoring/logs/train_52_scratch'

dataset_split_name = "train"
fc_conv_padding='VALID'


# TRAIN
batch_size = 16
num_threads = 5
learning_rate = 0.001
is_training = True
keep_prob6 = 0.5
keep_prob7 = 0.5
num_epochs = 100
num_batches = int(num_data_samples/batch_size)

# OPTIMIZER 
lr = 100
ce_lambda = 3
rmsprop_decay = 0.9
momentum = 0.9
opt_epsilon = 1.0
learning_rate_decay_type = 'exponential'
num_epochs_per_decay = 2.0
learning_rate_decay_factor =  0.94


def weight_variable(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape,name):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def fetch_images_valid(serialized_example, IMAGE_HEIGHT=224, IMAGE_WIDTH=224,is_training=False):
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

def fetch_images_train(serialized_example, IMAGE_HEIGHT=224, IMAGE_WIDTH=224,is_training=True):
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
    images, labels = tf.train.shuffle_batch( [preprocessed_image, label],
                                                     batch_size=batch_size,
                                                     capacity=batch_size+num_threads*batch_size,
                                                     num_threads=num_threads,
                                                     min_after_dequeue=batch_size)
    return images,labels

def get_data_files(dataset_split_name):
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


def configure_lr(global_step,learning_rate):
    decay_steps = int(num_data_samples / batch_size *
                    num_epochs_per_decay)
    return tf.train.exponential_decay(learning_rate,global_step,
                    decay_steps,learning_rate_decay_factor,
                    staircase=True,name='exponential_decay_learning_rate')


# placeholders
labels = tf.placeholder(dtype= tf.float32,shape=[batch_size,1])
images = tf.placeholder(dtype= tf.float32,shape=[batch_size,224,224,3])
keep_prob6 = tf.placeholder(dtype= tf.float32)
keep_prob7 = tf.placeholder(dtype= tf.float32)

sval_acc = tf.placeholder(dtype= tf.float32)
tf.summary.scalar('Validation Accuracy', sval_acc)
strain_acc = tf.placeholder(dtype= tf.float32)
tf.summary.scalar('Training Accuracy', strain_acc)
# sloss = tf.placeholder(dtype= tf.float32)
# tf.summary.scalar('Cross Entropy Loss', loss)

# Inference
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

with tf.variable_scope('vgg_16_new'):
  W_fc8 = weight_variable([1,1,4096,52],name="fc8/weights")
  b_fc8 = bias_variable([52],name="fc8/biases")
    
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

tf.logging.set_verbosity(tf.logging.INFO)

list_filenames = get_data_files('validation')
filename_queue = tf.train.string_input_producer(
     list_filenames, capacity=batch_size, shuffle=False )
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
[valid_images,valid_labels] = fetch_images_valid(serialized_example)
# valid_labels -= 1
valid_labels = tf.cast(valid_labels,tf.int64)
valid_labels = tf.squeeze(valid_labels)

# Input pipeline for train data
list_filenames_train = get_data_files('train')
filename_queue_train = tf.train.string_input_producer(
     list_filenames_train, capacity=batch_size, shuffle=True )
reader_train = tf.TFRecordReader()
_, serialized_example_train = reader_train.read(filename_queue_train)
[train_images,train_labels] = fetch_images_train(serialized_example_train)
# train_labels -= 1
train_labels = tf.cast(train_labels,tf.int64)
train_labels = tf.squeeze(train_labels)


labels = tf.cast(labels,tf.int64)
labels = tf.squeeze(labels)
lbl_one_hot = tf.one_hot(labels, 52, 1.0, 0.0)

prediction = tf.argmax(logits,1)
correct_prediction = tf.reduce_sum(tf.cast(tf.equal(prediction,labels),tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= lbl_one_hot, logits= logits)) 

# optimization
global_step = tf.Variable(0,name='global_step',trainable=False)
learning_rate = configure_lr(global_step,learning_rate)

old_vgg_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vgg_16")
new_vgg_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="vgg_16_new")
all_vars = old_vgg_params + new_vgg_params

optimizer = tf.train.RMSPropOptimizer(
            learning_rate/lr,decay=rmsprop_decay,
            momentum=momentum,epsilon=opt_epsilon)
train_step = optimizer.minimize(cross_entropy,global_step=global_step,var_list= all_vars)


init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

# graph ends ! you can create session now!
merged = tf.summary.merge_all()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init_op)
    list_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
    saver = tf.train.Saver(list_vars)
    saver.restore(sess, checkpoint)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)

    train_writer = tf.summary.FileWriter(summary_location, sess.graph)
    steps = num_batches*num_epochs

    train_instances = num_batches*batch_size
    val_instances = (int((52*50)/batch_size))*batch_size
    epoch_counter = 0
    best_Acc = 0

    train_acc = 0
    for i in tqdm(range(steps)):
        img,lbl = sess.run([train_images,train_labels])
        # lbl = np.reshape(lbl,[batch_size,1])
        loss, pred,true_lbl,c_pred,_ = sess.run([cross_entropy,prediction,labels,correct_prediction,train_step],feed_dict={labels:lbl, images:img, keep_prob6:0.7, keep_prob7:0.7})

        train_acc += c_pred

        if i%(num_batches-1) ==0 :
            valid_acc = 0
            for k in tqdm(range(int((52*50)/batch_size))):
                valid_img,valid_lbl = sess.run([valid_images,valid_labels])
                # valid_lbl = np.reshape(valid_lbl,[batch_size,1])
                acc = sess.run(correct_prediction,feed_dict={labels:valid_lbl, images:valid_img, keep_prob6:1.0, keep_prob7:1.0})
                valid_acc += acc
 
            epoch_val_acc = float(valid_acc)/val_instances
            epoch_train_acc = float(train_acc)/train_instances

            if epoch_val_acc > best_Acc:
                saver.save(sess,save_dir+'/model.ckpt')
                best_Acc = epoch_val_acc

            _merged = sess.run(merged, feed_dict={strain_acc: epoch_train_acc, sval_acc:epoch_val_acc})
            train_writer.add_summary(_merged, epoch_counter)
            epoch_counter += 1
            train_acc = 0

            print('epoch_train_acc: ', epoch_train_acc)
            print('epoch_train_acc: ', epoch_val_acc)
            print('Best val till: ', best_Acc)

    coord.request_stop()
    coord.join(threads)
