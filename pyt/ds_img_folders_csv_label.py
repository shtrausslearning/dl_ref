''' Custom Dataset '''

# images stored in folders

# data located in folder: data_dir, has subfolders train , valid
# images are stored in these two folders (train/valid)
# only a subset of indicies are chosen, set to 4000 (can remove idx_choose)

# labels

# image labels are stored in train_labels.csv
# pillow used to open image, return image and label with get special method

class pytorch_data(Dataset):
    
    def __init__(self,data_dir,transform,data_type="train"):      
    
        # Get Image File Names
        cdm_data=os.path.join(data_dir,data_type)  # directory of files
        
        file_names = os.listdir(cdm_data) # get list of images in that directory  
        idx_choose = np.random.choice(np.arange(len(file_names)), 
                                      4000,
                                      replace=False).tolist()
        file_names_sample = [file_names[x] for x in idx_choose]
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names_sample]   # get the full path to images
        
        # Get Labels
        labels_data=os.path.join(data_dir,"train_labels.csv") 
        labels_df=pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True) # set data frame index to id
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in file_names_sample]  # obtained labels from df
        self.transform = transform
      
    def __len__(self):
        return len(self.full_filenames) # size of dataset
      
    # open image, apply transforms and return with label
    def __getitem__(self, idx):
        
        # Open Image with PIL
        image = Image.open(self.full_filenames[idx])  
        
        # Apply Specific Transformation to Image to get tensor
        image = self.transform(image) 
        
        return image, self.labels[idx]
    
    
# define transformation that converts a PIL image into PyTorch tensors
import torchvision.transforms as transforms 
data_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((46,46))])

# Define an object of the custom dataset for the train folder
img_dataset = pytorch_data(data_dir, data_transformer, "train") 
