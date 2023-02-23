import torch
from torch.utils.data import Dataset
import os
import numpy as np
from app.trainer.data_util import load_image, build_label_dict

class FurnitureDataset(Dataset):

    def __init__(self, path, img_size=224):
        super().__init__()
        self.path = path
        self.img_size = img_size

        data = []
        label_dict = build_label_dict(self.path)
        self.num_class = len(label_dict)

        for folder_name in os.listdir(self.path):
            if os.path.isdir(os.path.join(self.path, folder_name)):
                label = label_dict[folder_name]
                for img_name in os.listdir(os.path.join(self.path, folder_name)):
                    img_path = os.path.join(self.path, folder_name, img_name)
                    img = load_image(img_path, self.img_size)
                    data.append((img, label))

        self.images = [x[0] for x in data]
        self.labels = np.array([x[1] for x in data])

    def __getitem__(self, index):
        data = self.images[index]
        return data, self.labels[index]

    def __len__(self):
        return len(self.images)

    def num_classes(self):
        return self.num_class
