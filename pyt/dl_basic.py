from torch.utils.data import DataLoader

# Training DataLoader
train_dl = DataLoader(ds_train,
                      batch_size=32, 
                      shuffle=True)

# Validation DataLoader
val_dl = DataLoader(ds_val,
                    batch_size=32,
                    shuffle=False)
