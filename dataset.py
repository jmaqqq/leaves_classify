import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

class LeavesDataset(Dataset):
    def __init__(self,dataframe,data_dir,class_to_idx,transform=None):
        self.dataframe=dataframe.reset_index(drop=True)
        self.data_dir=Path(data_dir)
        self.class_to_idx=class_to_idx
        self.transform=transform

    def __len__(self,):
        return len(self.dataframe)

    def __getitem__(self,idx):
        row=self.dataframe.iloc[idx]
        image_path=self.data_dir/row["image"]
        label=self.class_to_idx[row["label"]]

        image=Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image=self.transform(image)
        return image,label


def get_data_loaders(data_dir,batch_size=128,mode="baseline"):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    # test_df=pd.read_csv(data_dir/"test.csv")

    classes = sorted(train_df["label"].unique())
    class_to_idx = {}
    for idx, name in enumerate(classes):
        class_to_idx[name] = idx

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if mode=="baseline":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    elif mode=="aug":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"]
    )

    train_dataset = LeavesDataset(
        train_df,
        data_dir,
        class_to_idx,
        train_transform
    )

    val_dataset = LeavesDataset(
        val_df,
        data_dir,
        class_to_idx,
        val_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, classes, class_to_idx







