import datetime
import os.path

import dataset_build
import torch
import pytorch_lightning as pl
from model import MaskModel

aug_downscale_path = os.path.join("aug_dataset","downscale")
aug_flip_path = os.path.join("aug_dataset","flip")
aug_blur_path = os.path.join("aug_dataset","blur")
with_mask_path = os.path.join("dataset", "with_mask")
no_mask_path = os.path.join("dataset", "no_mask")

with_mask_dirs = [with_mask_path, os.path.join(with_mask_path, aug_downscale_path), os.path.join(with_mask_path, aug_flip_path), os.path.join(with_mask_path, aug_blur_path)]
no_mask_dirs = [no_mask_path, os.path.join(no_mask_path, aug_downscale_path), os.path.join(no_mask_path, aug_flip_path), os.path.join(no_mask_path, aug_blur_path)]

dataset = dataset_build.MaskDataset(with_mask_dirs, no_mask_dirs)

n_samples = len(dataset)
train_size = int(len(dataset) * 0.6)
val_size = int(len(dataset)*0.2)
test_size = n_samples - (train_size+val_size)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="model/",
    filename=f"bert_{str(datetime.datetime.today())}"
)



trainer = pl.Trainer(
    gpus=1,
    max_epochs=200,
    callbacks=[checkpoint, pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min")]
)

model = MaskModel(lr=1e-5)
trainer.fit(model, train_loader, val_loader)

test = trainer.test(model, test_loader)
print(test)

