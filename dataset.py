import torch
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision.transforms import ToPILImage, ToTensor, transforms
import os 
import pandas as pd
from torchvision.transforms.functional import crop 

class Pic_to_Pic_dataset(Dataset): 
    def __init__(self, data_csv, transform): 
        super().__init__() 
        self.df = pd.read_csv(data_csv).sort_values('patient_id')[:30000]
        self.df =  self.df.reset_index()
        self.transform = transform

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        img_path = '/home/shivac/qml-data/'+self.df.img_path[index]
        mask_path = '/home/shivac/qml-data/'+self.df.mask_path[index] 
        img = Image.open(img_path).convert('L') 
        mask = Image.open(mask_path)
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask


class Crop_dataset(Dataset): 
    def __init__(self, data_csv, transform): 
        super().__init__() 
        self.df = pd.read_csv(data_csv).sort_values('patient_id')[:30000]
        self.df =  self.df.reset_index()
        self.transform = transform

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        img_path = '/home/shivac/qml-data/'+self.df.img_path[index]
        mask_path = '/home/shivac/qml-data/'+self.df.mask_path[index] 
        xc = self.df.xc[index]
        yc = self.df.yc[index]
        xc += 100
        xmin = xc - 50 
        xmax = xc + 50 
        ymin = yc - 50 
        ymax = yc + 50 
        print(f'{index = }')
        print(f'{xmin, ymin, xmax, ymax = }')
        width = xmax - xmin 
        height = ymax - ymin
        img = Image.open(img_path).convert('L') 
        mask = Image.open(mask_path)
        img = self.transform(img)
        mask = self.transform(mask)
        ToPILImage()(mask).save('./samples/mask.png')
        img = img[:, xmin:xmax, ymin:ymax]
        mask = mask[:, xmin:xmax, ymin:ymax]
        return img, mask



class Cond_Pic_to_Pic_dataset(Dataset): 
    def __init__(self, data_csv): 
        super().__init__() 
        self.df = pd.read_csv(data_csv)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        index = 100
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
    import torchvision.transforms as T 
    transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip()])
    dataset = Crop_dataset('/home/shivac/qml-data/csv_files/crop_train_80.csv',
                                 transform=transform) 
    from torch.utils.data import DataLoader 
    loader = DataLoader(dataset, batch_size=2) 
    img, mask= next(iter(loader)) 
    print(img.shape, mask.shape)
    ToPILImage()(img[0]).save('./samples/im.png')
    ToPILImage()(mask[0]).save('./samples/crop_mask.png')
    

