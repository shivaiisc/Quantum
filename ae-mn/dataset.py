import torch
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision.transforms import ToTensor
import os 
import pandas as pd 

class Pic_to_Pic_dataset(Dataset): 
    def __init__(self, data_csv): 
        super().__init__() 
        self.df = pd.read_csv(data_csv)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        img_path = '/home/shivac/qml-data/'+self.df.img_path[index]
        mask_path = '/home/shivac/qml-data/'+self.df.mask_path[index] 
        img = Image.open(img_path).convert('L') 
        mask = Image.open(mask_path)
        img = ToTensor()(img)
        mask = ToTensor()(mask)
        return img, mask



class Cond_Pic_to_Pic_dataset(Dataset): 
    def __init__(self, data_csv): 
        super().__init__() 
        self.df = pd.read_csv(data_csv)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        img_path = '/home/shivac/qml-data/'+self.df.img_path[index]
        mask_path = '/home/shivac/qml-data/'+self.df.mask_path[index] 
        img_path = img_path.split('/')
        img_path[-2] = str(0) 
        img_path = '/'.join(img_path)
        idx = int(self.df.idx[index])
        img = Image.open(img_path).convert('L') 
        mask = Image.open(mask_path)
        img = ToTensor()(img)
        mask = ToTensor()(mask)
        return img, mask, torch.as_tensor(idx)


if __name__ == '__main__': 
    # dataset = Seq_Median_Nerve_Dataset(path='../.data/') 
    dataset = Cond_Pic_to_Pic_dataset('../../../qml-data/csv_files/train_80.csv') 
    from torch.utils.data import DataLoader 
    loader = DataLoader(dataset, batch_size=2) 
    img, mask, idx= next(iter(loader)) 
    print(img.shape, mask.shape, idx)
    

