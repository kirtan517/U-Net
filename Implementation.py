import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds 
from tensorflow import keras 

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(10)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

class Double_Conv2D(keras.layers.Layer):

  def __init__(self,filters,kernel,**kwargs):
    super().__init__(**kwargs)
    self.Conv1=keras.layers.Conv2D(filters,kernel,activation="relu",padding="same")
    self.Conv2=keras.layers.Conv2D(filters,kernel,activation="relu",padding="same")

  def call(self,inputs):
    Z=inputs
    Z=self.Conv1(Z)
    Z=self.Conv2(Z)
    return Z

class U_Net(keras.Model):

  def __init__(self,input_shape=(128,128,3),**kwargs):
    super().__init__(**kwargs)
    self.input1 = keras.layers.Input((128,128,3))
    
    self.doubleConv1=Double_Conv2D(16,(3,3),input_shape=(128,128))
    self.maxpool1=keras.layers.MaxPool2D(pool_size=(2,2),strides=2)

    self.doubleConv2 = Double_Conv2D(32, (3, 3))
    self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    self.doubleConv3 = Double_Conv2D(64, (3, 3))
    self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    self.doubleConv4 = Double_Conv2D(128, (3, 3))
    self.maxpool4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    self.doubleConv5=Double_Conv2D(256,(3,3))
    self.transpose5 = keras.layers.Conv2DTranspose(128,kernel_size=2, strides=(2, 2))

    self.doubleConv6 = Double_Conv2D(128,(3, 3))
    self.transpose6= keras.layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2))
    
    self.doubleConv7 = Double_Conv2D(64, (3, 3))
    self.transpose7= keras.layers.Conv2DTranspose(32, kernel_size=2, strides=(2, 2))

    self.doubleConv8 = Double_Conv2D(32, (3, 3))
    self.transpose8= keras.layers.Conv2DTranspose(16, kernel_size=2, strides=(2, 2))

    self.doubleConv9 = Double_Conv2D(16, (3, 3))

    self.output1=keras.layers.Conv2D(3,kernel_size=(3,3),padding="same")

  def call(self,inputs):
    Z=inputs
    
    Z1=self.doubleConv1(Z)
    Z2=self.maxpool1(Z1)

    Z2 = self.doubleConv2(Z2)
    Z3 = self.maxpool2(Z2)

    Z3 = self.doubleConv3(Z3)
    Z4 = self.maxpool3(Z3)

    Z4 = self.doubleConv4(Z4)
    Z5 = self.maxpool4(Z4)
    # up

    Z6=self.doubleConv5(Z5)
    Z7=self.transpose5(Z6)
    print(Z4.shape,Z7.shape)
    y=keras.layers.concatenate([Z4,Z7])

    Z8 = self.doubleConv6(y)
    Z9 = self.transpose6(Z8)
    print(Z3.shape, Z9.shape)
    y = keras.layers.concatenate([Z3, Z9])

    Z10 = self.doubleConv7(y)
    Z11 = self.transpose7(Z10)
    print(Z2.shape, Z11.shape)
    y = keras.layers.concatenate([Z2, Z11])

    Z12 = self.doubleConv8(y)
    Z13 = self.transpose8(Z12)
    print(Z1.shape, Z13.shape)
    y = keras.layers.concatenate([Z1, Z13])

    Z14 = self.doubleConv9(y)
    Z15=self.output1(Z14)
    return Z15

model= U_Net()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_dataset,epochs=5)
print(model.summary())




