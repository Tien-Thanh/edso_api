import cv2
from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model
app = Flask(__name__)

@app.route("/make-prediction", methods=["POST"])
def image_prediction():
    try:
        model = load_model("model/fashion_mnist_3.h5")
        filestr = request.files['image'].read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (28, 28))
        if gray[1][1] > 127:
            gray = cv2.bitwise_not(gray)
        gray = gray.reshape(1, 28, 28, 1)
        y = model.predict(gray)
        pre = int(np.argmax(y))
        final = num_to_string(num=pre)
        return final
    except:
        return "Fail to load file."

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


