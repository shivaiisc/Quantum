import torch
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision.transforms import ToTensor
import os 
import pandas as pd 

class Seq_Median_Nerve_Dataset(Dataset): 
    def __init__(self, path): 
        super().__init__()
        self.patients_path = [os.path.join(path, patients_path) 
            for patients_path in os.listdir(path)]

    def __len__(self, ): 
        return len(self.patients_path)

    def __getitem__(self, index):
        path = self.patients_path[index]
        img, mask = sorted(os.listdir(path))
        img = Image.open(os.path.join(path, img)).convert('L')
        img = ToTensor()(img)
        mask = torch.load(os.path.join(path, mask)).squeeze()
        return img, mask 

class Pic_to_Pic_dataset(Dataset): 
    def __init__(self, data_csv): 
        super().__init__() 
        self.df = pd.read_csv(data_csv).sort_values(by='patient_id')[:30000].reset_index()

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
        self.df = pd.read_csv(data_csv).sort_values(by='patient_id')[:30000].reset_index()

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        img_path = '/home/shivac/qml-data/'+self.df.img_path[index]
        mask_path = '/home/shivac/qml-data/'+self.df.mask_path[index] 
        idx = int(self.df.idx[index])
        img = Image.open(img_path).convert('L') 
        mask = Image.open(mask_path)
        img = ToTensor()(img)
        mask = ToTensor()(mask)
        return img, mask, torch.as_tensor(idx)


if __name__ == '__main__': 
    # dataset = Seq_Median_Nerve_Dataset(path='../.data/') 
    dataset = Cond_Pic_to_Pic_dataset('../../qml-data/qml_mns.csv') 
    from torch.utils.data import DataLoader 
    loader = DataLoader(dataset, batch_size=2) 
    img, mask, idx= next(iter(loader)) 
    print(img.shape, mask.shape, idx)
    

