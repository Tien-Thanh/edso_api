import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, InputLayer, BatchNormalization, Dropout
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
import cv2

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_train = utils.to_categorical(y_train, 10)
Y_val = utils.to_categorical(y_val, 10)
Y_test = utils.to_categorical(y_test, 10)


model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(BatchNormalization())

model.add(Conv2D(64, (4, 4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.1))

model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=256, epochs=20, verbose=1)

model.save('fashion_mnist_3.h5')

numOfEpoch = 20
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()

pred = model.predict(X_test)
pred_digits = np.argmax(pred, axis=1)
print(accuracy_score(Y_test, pred_digits))