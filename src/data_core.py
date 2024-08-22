import time
import sys
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from PIL import Image


class myDataset(Dataset):
    def __init__(self, list_img=None):
        self.tamp_list = list_img
    def shuffle(self):
        random.shuffle(self.tamp_list)
    def __len__(self):
        return len(self.tamp_list)

    # def __getitem__(self, index):
    #     assert self.tamp_list
    #     assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
    #     rgb_path = self.tamp_list[index]
    #     img_RGB = np.array(Image.open(rgb_path).convert("RGB"))
    #     return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path
    def __getitem__(self, index):
        assert self.tamp_list
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        rgb_path = self.tamp_list[index]
        
        start_time = time.time()
        img_RGB = np.array(Image.open(rgb_path).convert("RGB"))
        end_time = time.time()
        
        print(f"Time to load {rgb_path}: {end_time - start_time:.2f} seconds")
        sys.stdout.flush()
        return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path

    def get_filename(self, index):
        item = self.tamp_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item

