import os
import cv2
import numpy as np
import torch


def load_image(img_path, img_size):
    # img = torchvision.io.read_image(path=img_path)
    # img = T.Resize((img_size, img_size))(img)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = torch.from_numpy(np.array(img))
    data = img.reshape(3, img_size, img_size)
    data = data.float()
    return data


def build_label_dict(img_path):
    label_dict = {}
    num_class = 0
    for folder_name in os.listdir(img_path):
        if str(folder_name) in label_dict:
            pass
        label_dict[str(folder_name)] = num_class
        num_class += 1

    return label_dict
