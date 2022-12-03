from __future__ import division, print_function

import sys

import flask

# import dataset
import torch
from flask_cors import CORS
from torchvision import transforms

sys.path.append("../")
from concept_model.concept_model import (
    initialize_classify_model,
    initialize_concept_model,
    predict_class_model,
    predict_concept_model,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "CUB_200_2011/images"
concept_model_name = "inception"
num_concepts = 312
num_classes = 200
pic2attribute_batch_size = 36
attribute2class_batch_size = 128
pic2attribute_num_epochs = 25
attribute2class_num_epochs = 50
feature_extract = True

model_pic2concept, input_size = initialize_concept_model(
    concept_model_name, num_concepts, feature_extract
)
model_concept2class = initialize_classify_model(num_concepts, num_classes)

model_pic2concept = model_pic2concept.to(device)
model_concept2class = model_concept2class.to(device)

data_transform = transforms.Compose(
    [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model_pic2concept.load_state_dict(torch.load("../train/pic2concept.pth"))
model_concept2class.load_state_dict(torch.load("../train/concept2class.pth"))

model_pic2concept.eval()
model_concept2class.eval()

app = flask.Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", methods=["GET"])
def index():
    with open("index.html", "r") as f:
        return f.read(), 200


@app.route("/predictAttributes", methods=["POST"])
def predict_attributes():
    image = flask.request.form["image"]
    image = image[22:]

    attributes = predict_concept_model(model_pic2concept, image, data_transform, device)
    attribute_indices = list(range(1, 313))
    with open("../attributes.txt", "r") as f:
        attribute_names = [line.split(" ")[1] for line in f.readlines()]

    zipped = zip(attribute_indices, attribute_names, attributes)
    zipped = sorted(zipped, key=lambda x: x[2], reverse=True)
    attribute_indices, attribute_names, attributes = zip(*zipped)

    return {
        "attributes": list(attributes),
        "attributeIndices": list(attribute_indices),
        "attributeNames": list(attribute_names),
    }, 200


@app.route("/predictClass", methods=["POST"])
def predict_class():
    attributes = flask.request.get_json()["attributes"]
    attribute_indices = flask.request.get_json()["attributeIndices"]
    zipped = zip(attribute_indices, attributes)
    zipped = sorted(zipped, key=lambda x: x[0])
    attribute_indices, attributes = zip(*zipped)

    classes = predict_class_model(model_concept2class, attributes, device)
    class_indices = list(range(1, 201))
    with open("../CUB_200_2011/classes.txt", "r") as f:
        class_names = [line.split(" ")[1] for line in f.readlines()]

    zipped = zip(class_indices, class_names, classes)
    zipped = sorted(zipped, key=lambda x: x[2], reverse=True)
    class_indices, class_names, classes = zip(*zipped)

    return {
        "classes": list(classes),
        "classIndices": list(class_indices),
        "classNames": list(class_names),
    }, 200
