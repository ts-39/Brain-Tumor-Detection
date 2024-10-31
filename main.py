import cv2
import os
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import math


def get_jpg_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]

def get_images(directory_path):
    list_image = []
    list_path = get_jpg_paths(directory_path)

    for i in list_path:
        im_gray = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, dsize=(256,256))
        list_image.append(im_gray)

    return list_image

yes_list = get_images('brain_tumor_dataset/yes')
no_list = get_images('brain_tumor_dataset/no')
x_data = yes_list + no_list

x_data = np.array(x_data).astype("float32")

y_data_yes = np.tile(1, len(yes_list))
y_data_no = np.tile(0, len(no_list))
y_data = np.concatenate((y_data_yes, y_data_no)).astype("uint8")

x_data /= 255

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# #---------------MODEL DESIGN---------------
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(256,256, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# model.summary()

epochs = 20
batch_size = 32
start_time = time.time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Computation time:{0:.3f} sec'.format(time.time() - start_time))

# SCORE1
# Test loss: 0.8309130072593689
# Test accuracy: 0.9019607901573181
# Computation time:158.185 sec
# epochs = 20

# SCORE2
# Test loss: 0.36524829268455505
# Test accuracy: 0.8627451062202454
# Computation time:49.700 sec
# epochs = 6

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig("history3.png") 
plt.show()

predict = model.predict(x_test, batch_size=32)
pre_ans = predict.argmax(axis=1)

# y_test_change = np.argmax(y_test, axis=1) #onehotを普通のにもどす
a = y_test == pre_ans


false_list = []
for i in range(len(a)):
    item = a[i]
    if item == False:
        false_list.append(i)

false_pic_list = []
for i in false_list:
    false_pic_list.append(x_test[i])

print(len(false_pic_list))

# print(type(false_pic_list))
false_pic = np.array(false_pic_list)
# print(type(false_pic))

import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image

#画像表示
def ConvertToImg(img):
    return Image.fromarray(np.uint8(img))


chr_w = 256
chr_h = 256
num = math.floor(math.sqrt(len(false_pic_list)))

canvas = Image.new('RGB', (int(chr_w * num/2), int(chr_h * num/2)), (255, 255, 255))

i = 0
for y in range( int(num/2) ):
    for x in range( int(num/2) ):
        chrImg = ConvertToImg(false_pic[i].reshape(chr_w, chr_h))
        canvas.paste(chrImg, (chr_w*x, chr_h*y))
        i = i + 1
    

canvas.show()
# 表示した画像をJPEGとして保存
# canvas.save('mnist.jpg', 'JPEG', quality=100, optimize=True)