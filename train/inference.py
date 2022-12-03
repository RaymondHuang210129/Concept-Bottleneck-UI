from __future__ import division, print_function

import sys

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

sys.path.append("../")
from concept_model.concept_model import (
    initialize_classify_model,
    initialize_concept_model,
    validate_concept_model,
)
from cub2011.cub2011 import Cub2011

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

pic2class_datasets = {
    x: Cub2011(
        "../",
        train=True,
        transform=data_transform,
        download=True,
        mode="pic2class",
    )
    for x in ["train", "val"]
}

pic2class_dataloader = torch.utils.data.DataLoader(
    pic2class_datasets["val"],
    batch_size=pic2attribute_batch_size,
    shuffle=True,
    num_workers=4,
)

model_pic2concept.load_state_dict(torch.load("pic2concept.pth"))
model_concept2class.load_state_dict(torch.load("concept2class.pth"))

validate_concept_model(
    model_pic2concept,
    model_concept2class,
    pic2class_dataloader,
    nn.CrossEntropyLoss(),
    device,
)
