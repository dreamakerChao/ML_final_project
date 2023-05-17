from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#data
from PIL import Image
import os
#%%
x_train = []
x_test=[]
classes = 5999
path = "E:/ML_project/German/"

df = pd.read_csv(path+'train.csv')
y_train=df.iloc[:,6]
file_name=df.iloc[:,7]
for i in range(39209):
    new_path=path+file_name[i]
    #print(new_path)
    image=Image.open(new_path)
    image = image.resize((128,128))
    image = np.array(image)
    #sim = Image.fromarray(image)
    x_train.append(image)
    print('\r' + '{:.2%}'.format(i/39209), end='', flush=True)

df = pd.read_csv(path+'test.csv')
y_test=df.iloc[:,6]
file_name=df.iloc[:,7]
print('\n')

for i in range(12630):
    new_path=path+file_name[i]
    #print(new_path)
    image=Image.open(new_path)
    image = image.resize((128,128))
    image = np.array(image)
    #sim = Image.fromarray(image)
    x_test.append(image)
    print('\r' + '{:.2%}'.format(i/12630), end='', flush=True)
#%%%
print('\n')
x_train=np.array(x_train)
x_test=np.array(x_test)
x_train = x_train.reshape(39209, 128, 128, 3).astype('float32')
x_test = x_test.reshape(12630, 128, 128, 3).astype('float32')
x_train = x_train / 255
print('x_train compelte\n')
x_test = x_test / 255
print('x_test compelte\n')
y_train = np_utils.to_categorical(y_train) #one-hot
y_test_categories = y_test
y_test = np_utils.to_categorical(y_test)

#%%
def plot_out(history):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    

def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(43))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
    model.summary()

    history =model.fit(x_train, y_train, batch_size=400, epochs=5)
    test_loss, testacc = model.evaluate(x_train, y_train)
    print("Finished training:", testacc)
    model.save('410987011_model.h5')
    return history,model
#%%
if __name__=='__main__':
    history,model=get_model()
    plot_out(history)
    model.evaluate(x_test,y_test)
#%%
