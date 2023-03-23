''' Splitting read dataset into tts '''
# requires loaded dataset (eg. ImageFolder)

img_dataset = ImageFolder('.data',transform = transforms.Compose([transforms.ToTensor()])

# Training and validation splits
# torch has random_split (torch.utils.data)

from torch.utils.data import random_split

len_img=len(img_dataset)
len_train=int(0.8*len_img)
len_val=len_img-len_train

# Split Pytorch tensor
ds_train,ds_val = random_split(img_dataset,
                             [len_train,len_val]) # random split 80/20
                          
                          
# Define the following transformations for the training dataset
tr_transf = transforms.Compose([transforms.Resize((40,40)),
                                transforms.ToTensor()])

# For the validation dataset, we don't need any augmentation; simply convert images into tensors
val_transf = transforms.Compose([transforms.ToTensor()])

# After defining the transformations, overwrite the transform functions of train_ts, val_ts
ds_train.transform=tr_transf
ds_val.transform=val_transf

# and 

# sklearn train_test_split may be more intuitive
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def train_val_ds(dataset,split=0.2):
  train_idx, val_idx  train_test_split(list(range(len(dataset))),
                                       test_size = split )
  
  datasets = {}
  datasets['train'] = Subset(dataset,train_idx)
  dataset['val'] = Subset(dataset,val_idx)
  return datasets
                          
datasets = train_val_ds(img_dataset)
ds_train = datasets['train']
ds_val = datasets['val']

# some dataset methods

dataset.classes
dataset.class_to_idx
dataset.imgs


