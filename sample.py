from flask import Flask
from flask_cors import cross_origin
from flask_restful import Api, Resource, reqparse
import werkzeug
from tensorflow.keras.models import load_model
import numpy as np
import cv2


# create class to make prediction
class ImageFileChecker():
    def __init__(self):
        self.model = load_model("fashion_mnist_3.h5")

    def check(self, file_bytes):
        try:

            npimg = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (28,28))
            if gray[1][1] > 127:
                gray = cv2.bitwise_not(gray)
            gray = gray.reshape(1, 28, 28, 1)
            y = self.model.predict(gray)
            pre = int(np.argmax(y))
            fila = self.num_to_string(num=pre)
            return fila

        except:
            img = None
        if img is None:
            return "Can not load image."

    def num_to_string(self, num):
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


# create service to handle HTTP request
class CheckImageFileService(Resource):
    def __init__(self, parser: reqparse.RequestParser, checker: ImageFileChecker):
        self.parser = parser
        self.checker = checker

    @cross_origin(supports_credentials= True)
    def get(self):
        return "Please use POST method."

    @cross_origin(supports_credentials= True)
    def post(self):
        args = self.parser.parse_args()
        file = args.file
        prediction = 0
        if file:
            file_bytes = file.read()
            prediction = self.checker.check(file_bytes)
        return {"Image Prediction": prediction}


# create an app and route resources
def create_app():
    app = Flask(__name__)
    api = Api(app)
    parser = reqparse.RequestParser()
    parser.add_argument(
        "file",
        required=True,
        type=werkzeug.datastructures.FileStorage,
        location="files",
    )
    checker = ImageFileChecker()
    api.add_resource(
        CheckImageFileService,
        "/make-prediction",
        resource_class_kwargs={"parser": parser, "checker": checker},
    )
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3012)



