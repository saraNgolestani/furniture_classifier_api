import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
import numpy as np


class FurnitureDataset(Dataset):

    def __init__(self, path, img_size=224):
        super().__init__()
        self.path = path
        self.img_size = img_size
        self.resize_transform = T.Resize((self.img_size, self.img_size))

        data = []
        label_dict = {}
        self.num_class = 0
        for folder_name in os.listdir(self.path):
            if str(folder_name) in label_dict:
                pass
            label_dict[str(folder_name)] = self.num_class
            self.num_class += 1
            if os.path.isdir(os.path.join(self.path, folder_name)):
                label = label_dict[folder_name]
                for img_name in os.listdir(os.path.join(self.path, folder_name)):
                    img_path = os.path.join(self.path, folder_name, img_name)
                    img = torchvision.io.read_image(path=img_path)
                    img = self.resize_transform(img)
                    data.append((img, label))

        # Convert the list into numpy arrays
        self.images = np.array([x[0] for x in data])
        self.labels = np.array([x[1] for x in data])

    def __getitem__(self, index):
        data = self.images[index].reshape(3, 224, 224)
        data = data.float()
        return data, self.labels[index]

    def __len__(self):
        return len(self.images)

    def num_classes(self):
        return self.num_class
