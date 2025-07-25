import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class GenetypeDataset(Dataset):
    def __init__(self, ppg, label, hr, mean_ph):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.ppg = ppg
        self.hr = hr
        # self.weights = weights
        self.label = label
        self.mean_ph = mean_ph

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        #print(len(self.ppg[index]))
        ppg = torch.tensor(self.ppg[index], dtype=torch.float32)
        #print("ppg",ppg.shape)
        targets = torch.tensor(self.hr[index], dtype=torch.float32)-self.mean_ph
        # weights = torch.tensor(self.hr[index], dtype=torch.float32)
        label = torch.tensor(self.hr[index], dtype=torch.float32)
        inputs = ppg
        return inputs, targets, label
        # return inputs, targets, weights, label

    def __len__(self):
        return len(self.ppg)