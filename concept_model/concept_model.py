from __future__ import division, print_function

import base64
import copy
import io
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.ops import MLP


def train_concept_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_concepts,
    num_epochs=25,
    is_inception=False,
    device="cpu",
):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == "train":
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    preds = torch.sigmoid(outputs) > 0.5
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels.data).sum().double() / num_concepts

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_concept_model(
    model_name, num_classes, feature_extract, use_pretrained=True
):
    model_ft = None
    input_size = 0

    model_ft = models.inception_v3(
        pretrained=use_pretrained,
    )
    set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299

    return model_ft, input_size


def initialize_classify_model(num_concepts, num_classes):
    return MLP(in_channels=num_concepts, hidden_channels=[num_classes]).double()


def train_class_model(
    model, dataloaders, criterion, optimizer, num_epochs=1000, device="cpu"
):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs.double())
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "val":
                val_acc_history.append(epoch_acc)
        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def validate_concept_model(
    concept_model, classify_model, dataloader, classify_criterion, device
):

    running_loss = 0.0
    running_corrects = 0.0

    for pics, labels in dataloader:
        pics = pics.to(device)
        labels = labels.to(device)
        attributes, _ = concept_model(pics)
        attributes = torch.sigmoid(attributes)
        outputs = classify_model(attributes.double())
        loss = classify_criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * pics.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    print("val Loss: {:.4f} Acc: {:.4f}".format(total_loss, total_acc))


def predict_concept_model(concept_model, pic, transform, device) -> list:
    pic = Image.open(io.BytesIO(base64.b64decode(pic)))
    pic = transform(pic).unsqueeze(0).to(device)
    attributes = concept_model(pic)
    attributes = torch.sigmoid(attributes)
    return attributes.tolist()[0]


def predict_class_model(classify_model, attributes, device) -> list:
    attributes = torch.tensor(attributes).unsqueeze(0).to(device)
    outputs = classify_model(attributes.double())
    outputs = torch.softmax(outputs, dim=1)
    return outputs.tolist()[0]
