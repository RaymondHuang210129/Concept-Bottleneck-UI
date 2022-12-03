# This file is based on the code from the following repository:
# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
# and further modified to fit the needs of this project.
import os
import tarfile
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url


class Cub2011(Dataset):
    base_folder = "CUB_200_2011/images"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        loader=default_loader,
        download=True,
        mode: Literal["pic2attributes", "attribute2class", "pic2class"] = "pic2class",
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.mode = mode

        if download:
            self._download()

        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )
        self.image_attribute_labels = pd.read_csv(
            os.path.join(
                self.root, "CUB_200_2011", "attributes", "image_attribute_labels.txt"
            ),
            delimiter=r"\s+",
            usecols=(0, 1, 2),
            names=["img_id", "attribute_id", "is_present"],
        )

        self.attributes_continuous = {}
        with open(
            os.path.join(
                self.root,
                "CUB_200_2011",
                "attributes",
                "class_attribute_labels_continuous.txt",
            )
        ) as f:
            class_num = 1
            for line in f:
                self.attributes_continuous[class_num] = torch.tensor(
                    [float(x) / 100 for x in line.split()]
                )
                class_num += 1

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        self.attributes = {
            sample["img_id"]: torch.tensor(
                self.image_attribute_labels[
                    self.image_attribute_labels["img_id"] == (sample["img_id"])
                ]["is_present"]
                .astype(float)
                .values
            )
            for sample in self.data.iloc
        }

    def _download(self):

        if os.path.exists(os.path.join(self.root, "CUB_200_2011")):
            print("Files already downloaded and verified")
            return
        print("downloading")

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.mode == "pic2attributes":
            return (
                img,
                self.attributes[sample["img_id"]],
            )
        elif self.mode == "attribute2class":
            return (
                self.attributes[sample["img_id"]],
                target - 1,
            )
        elif self.mode == "pic2attributes_continuous":
            return (
                img,
                self.attributes_continuous[target],
            )
        else:
            return img, target - 1

    def get_weights(self):
        s = torch.stack([x for _, x in self.attributes.items()], dim=0).sum(dim=0)
        return len(self.attributes) / s


if __name__ == "__main__":
    dataset = Cub2011(
        root="../",
        train=False,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        download=False,
        mode="pic2attributes",
    )
    sample = dataset.data.iloc[0]
    print(sample)
    print(dataset.attributes[sample["img_id"]].nonzero())
