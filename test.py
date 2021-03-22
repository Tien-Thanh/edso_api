import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("fashion_mnist_3.h5")

img = cv2.imread("\\Users\\Admin\\Desktop\\pullover.jpg")
img = cv2.resize(img, (28,28))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(gray.shape)
if gray[1][1] > 127:
    gray = cv2.bitwise_not(gray)
input = gray.reshape(28, 28, 1)
print(input.shape)

def num_to_string(num):
    prediction = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandals",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boots",
    }
    return prediction.get(num, "Fail to make prediction.")


y = model.predict(input.reshape(1, 28, 28, 1))
pre = int(np.argmax(y))
print("Du doan: ", num_to_string(pre))