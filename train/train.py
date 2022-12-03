from __future__ import division, print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

sys.path.append("../")
from concept_model.concept_model import (
    initialize_classify_model,
    initialize_concept_model,
    train_class_model,
    train_concept_model,
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
pic2attribute_num_epochs = 15
attribute2class_num_epochs = 100
feature_extract = True

model_pic2concept, input_size = initialize_concept_model(
    concept_model_name, num_concepts, feature_extract
)

print(model_pic2concept)

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

print("Initializing Datasets and Dataloaders...")

pic2attribute_datasets = {
    x: Cub2011(
        "../",
        train=True,
        transform=data_transforms["train"],
        download=True,
        mode="pic2attributes",
    )
    for x in ["train", "val"]
}

pic2attribute_dataloader = {
    x: torch.utils.data.DataLoader(
        pic2attribute_datasets[x],
        batch_size=pic2attribute_batch_size,
        shuffle=True,
        num_workers=4,
    )
    for x in ["train", "val"]
}

model_pic2concept = model_pic2concept.to(device)

pic2concept_params_to_update = model_pic2concept.parameters()
print("Params to learn:")
if feature_extract:
    pic2concept_params_to_update = []
    for name, param in model_pic2concept.named_parameters():
        if param.requires_grad == True:
            pic2concept_params_to_update.append(param)
            print("\t", name)

else:
    for name, param in model_pic2concept.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

pic2concept_optimizer_ft = optim.SGD(pic2concept_params_to_update, lr=1.0, momentum=0.9)
pic2concept_criterion = nn.BCEWithLogitsLoss()

try:
    model_pic2concept.load_state_dict(torch.load("pic2concept.pth"))
except FileNotFoundError:
    model_pic2concept, hist = train_concept_model(
        model_pic2concept,
        pic2attribute_dataloader,
        pic2concept_criterion,
        pic2concept_optimizer_ft,
        num_concepts,
        num_epochs=pic2attribute_num_epochs,
        is_inception=True,
        device=device,
    )
    torch.save(model_pic2concept.state_dict(), "pic2concept.pth")


model_concept2class = initialize_classify_model(num_concepts, num_classes)

attribute2label_datasets = {
    x: Cub2011(
        "../",
        train=True,
        transform=data_transforms["train"],
        download=True,
        mode="attribute2class",
    )
    for x in ["train", "val"]
}

attribute2label_dataloader = {
    x: torch.utils.data.DataLoader(
        attribute2label_datasets[x],
        batch_size=attribute2class_batch_size,
        shuffle=True,
        num_workers=4,
    )
    for x in ["train", "val"]
}


model_concept2class = model_concept2class.to(device)

print(model_concept2class)

concept2class_param_to_update = model_concept2class.parameters()
print("Params to Learn:")
for name, param in model_concept2class.named_parameters():
    if param.requires_grad == True:
        print("\t", name)

concept2class_optimizer_ft = optim.SGD(
    concept2class_param_to_update, lr=0.1, momentum=0.9
)
concept2class_criterion = nn.CrossEntropyLoss()

try:
    model_concept2class.load_state_dict(torch.load("concept2class.pth"))
except FileNotFoundError:
    model_concept2class, hist = train_class_model(
        model_concept2class,
        attribute2label_dataloader,
        concept2class_criterion,
        concept2class_optimizer_ft,
        num_epochs=attribute2class_num_epochs,
        device=device,
    )
    torch.save(model_concept2class.state_dict(), "concept2class.pth")
