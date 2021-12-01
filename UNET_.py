import numpy as np
import pandas as pd
import os
import random
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

#Code taken from https://www.kaggle.com/binh234/unet-segmentation

#Paramaters of input data
EPOCHS=20
BATCH_SIZE=32
HEIGHT=256
WIDTH=256
CHANNELS=3
N_CLASSES=13
AUTO = tf.data.AUTOTUNE

tf.test.is_gpu_available()

def loadImage(path):
    img = Image.open(path)
    img = np.array(img)

    image = img[:,:256]
    image = image / 255.0
    mask = img[:,256:]

    return image, mask

#uses bins to label data rather than other examples i found that use k means clustering.
def bin_image(mask):
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    return new_mask

def getSegmentationArr(image, classes, width=WIDTH, height=HEIGHT):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, : , 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c ).astype(int)
    return seg_labels

def give_color_to_seg_img(seg, n_classes=N_CLASSES):

    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)


train_folder = "../input/cityscapes-image-pairs/cityscapes_data/train"
valid_folder = "../input/cityscapes-image-pairs/cityscapes_data/val"
train_filenames = glob.glob(os.path.join(train_folder, "*.jpg"))
valid_filenames = glob.glob(os.path.join(valid_folder, "*.jpg"))

num_of_training_samples = len(train_filenames)
num_of_valid_samples = len(valid_filenames)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, filenames,
                 batch_size=BATCH_SIZE,
                 shuffle=True):

        self.filenames = filenames
        self.batch_size = BATCH_SIZE
        self.shuffle= shuffle
        self.n = len(self.filenames)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __get_data(self, batches):
        imgs=[]
        segs=[]
        for file in batches:
            image, mask = loadImage(file)
            mask_binned = bin_image(mask)
            labels = getSegmentationArr(mask_binned, N_CLASSES)
            labels = np.argmax(labels, axis=-1)

            imgs.append(image)
            segs.append(labels)

        return np.array(imgs), np.array(segs)

    def __getitem__(self, index):

        batches = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__get_data(batches)

        return (X, y)

    def __len__(self):

        return self.n // self.batch_size


train_gen = DataGenerator(train_filenames)
val_gen = DataGenerator(valid_filenames)

for imgs, segs in train_gen:
    break
imgs.shape, segs.shape

image = imgs[0]
mask = give_color_to_seg_img(segs[0])
masked_image = image * 0.5 + mask * 0.5

# fig, axs = plt.subplots(1, 3, figsize=(20,20))
# axs[0].imshow(image)
# axs[0].set_title('Original Image')
# axs[1].imshow(mask)
# axs[1].set_title('Segmentation Mask')
# #predimg = cv2.addWeighted(imgs[i]/255, 0.6, _p, 0.4, 0)
# axs[2].imshow(masked_image)
# axs[2].set_title('Masked Image')
# plt.show()

def unet():
    main_input = Input(shape=(HEIGHT, WIDTH, CHANNELS), name = 'img_input')

    ''' ~~~~~~~~~~~~~~~~~~~ ENCODING LAYERS ~~~~~~~~~~~~~~~~~~~ '''

    c1 = Conv2D(32, kernel_size=(3,3), padding = 'same')(main_input)
    c1 = LeakyReLU(0.2)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, kernel_size=(3,3), padding = 'same')(c1)
    c1 = LeakyReLU(0.2)(c1)
    c1 = BatchNormalization()(c1)

    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(p1)
    c2 = LeakyReLU(0.2)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(c2)
    c2 = LeakyReLU(0.2)(c2)
    c2 = BatchNormalization()(c2)

    p2 = MaxPooling2D((2,2))(c2)

    c3 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(p2)
    c3 = LeakyReLU(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(32*2, kernel_size=(1,1), padding = 'same')(c3)
    c3 = LeakyReLU(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(c3)
    c3 = LeakyReLU(0.2)(c3)
    c3 = BatchNormalization()(c3)

    p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(p3)
    c4 = LeakyReLU(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(32*4, kernel_size=(1,1), padding = 'same')(c4)
    c4 = LeakyReLU(0.2)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(c4)
    c4 = LeakyReLU(0.2)(c4)
    c4 = BatchNormalization()(c4)

    p4 = MaxPooling2D((2,2))(c4)

    c5 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(p4)
    c5 = LeakyReLU(0.2)(c5)
    c5 = BatchNormalization()(c5)


    ''' ~~~~~~~~~~~~~~~~~~~ DECODING LAYERS ~~~~~~~~~~~~~~~~~~~ '''

    u1 = UpSampling2D((2,2))(c5)
    concat1 = concatenate([c4, u1])

    c6 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(concat1)
    c6 = LeakyReLU(0.2)(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(c6)
    c6 = LeakyReLU(0.2)(c6)
    c6 = BatchNormalization()(c6)


    u2 = UpSampling2D((2,2))(c6)
    concat2 = concatenate([c3, u2])

    c7 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(concat2)
    c7 = LeakyReLU(0.2)(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(c7)
    c7 = LeakyReLU(0.2)(c7)
    c7 = BatchNormalization()(c7)

    u3 = UpSampling2D((2,2))(c7)
    concat3 = concatenate([c2, u3])

    c8 = Conv2D(32, kernel_size=(3,3), padding = 'same')(concat3)
    c8 = LeakyReLU(0.2)(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(32, kernel_size=(3,3), padding = 'same')(c8)
    c8 = LeakyReLU(0.2)(c8)
    c8 = BatchNormalization()(c8)

    u4 = UpSampling2D((2,2))(c8)
    concat4 = concatenate([c1, u4])

    c9 = Conv2D(16, kernel_size = (1,1), padding = 'same')(concat4)
    c9 = LeakyReLU(0.2)(c9)
    c9 = BatchNormalization()(c9)

    mask_out = Conv2D(N_CLASSES, (1,1), padding = 'same', activation = 'sigmoid', name = 'mask_out')(c9)

    model = Model(inputs = [main_input], outputs = [mask_out])

    return model

model = unet()
model.summary()


TRAIN_STEPS = len(train_gen)
VAL_STEPS = len(val_gen)

learning_rate = 0.001
decay_rate = learning_rate * 0.9
opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)

checkpoint = ModelCheckpoint('seg_model_v2.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
history2 = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=TRAIN_STEPS,
          validation_steps=VAL_STEPS, epochs=EPOCHS, callbacks = [checkpoint])


# loss = history2.history["val_loss"]
# acc = history2.history["val_accuracy"]
#
# plt.figure(figsize=(12, 10))
# plt.subplot(211)
# plt.title("Val. cce Loss")
# plt.plot(loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
#
# plt.subplot(212)
# plt.title("Val. Accuracy")
# plt.plot(acc)
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
#
# plt.tight_layout()
# plt.show()

model.load_weights("./seg_model_v2.h5")
test_gen = DataGenerator(valid_filenames, 1)

# test_iter = iter(test_gen)
# for i in range(12):
#     imgs, segs = next(test_iter)
#     pred = model.predict(imgs)
#     _p = give_color_to_seg_img(np.argmax(pred[0], axis=-1))
#     _s = give_color_to_seg_img(segs[0])
#
#     predimg = cv2.addWeighted(imgs[0], 0.5, _p, 0.5, 0)
#     trueimg = cv2.addWeighted(imgs[0], 0.5, _s, 0.5, 0)
#
#     plt.figure(figsize=(12, 4))
#     plt.subplot(131)
#     plt.title("Original")
#     plt.imshow(imgs[0])
#     plt.axis("off")
#
#     plt.subplot(132)
#     plt.title("Prediction")
#     plt.imshow(predimg)
#     plt.axis("off")
#
#     plt.subplot(133)
#     plt.title("Ground truth")
#     plt.imshow(trueimg)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig("pred_"+str(i)+".png", dpi=150)
#     plt.show()
