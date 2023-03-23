# torch.utils.data.dataset.random_split returns a Subset object which has no transforms attribute.
# Create dataset from subset types 

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split,TensorDataset

class MyDataset(Dataset):
    
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return (x, y)

    
# Dataset (random)
# 100 samples 3 layers 24x24
# 100 samples [0-10] labels 

dataset = TensorDataset(
    torch.randn(100, 3, 24, 24),
    torch.randint(0, 10, (100,)))

lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)]

# Split Dataset into Two Groups -> Subset Type w/o Transformations

ds_train, ds_valid = random_split(dataset, lengths)

transforms = transforms.Normalize((0.0, 0.0, 0.0),
                                  (0.5, 0.5, 0.5))

print(type(ds_train)) # class 'torch.utils.data.dataset.Subset'>

# Define Custom Dataset

datasetA = MyDataset(subsetA,transforms) # Create a Dataset using Subset 1
datasetB = MyDataset(subsetB,transforms)  # Create a Dataset using Subset 2
print(type(datasetA)) # <class '__main__.MyDataset'> # custom dataset w/ transform
