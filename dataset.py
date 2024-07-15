import numpy as np 
from einops import rearrange
import torch
from torch.utils.data import Dataset 
from PIL import Image 
from torchvision.transforms import ToPILImage, ToTensor, transforms
from torchvision.transforms import v2 
import pandas as pd
import albumentations as A
import cv2 

class Pic_to_Pic_dataset(Dataset): 
    def __init__(self, data_csv, mode='train'): 
        super().__init__() 
        self.mode=mode
        if mode == 'train':  
            self.df = pd.read_csv(data_csv).sort_values('patient_id')[:30000]
            self.df =  self.df.reset_index()
            self.transform = v2.Compose([v2.RandomHorizontalFlip(),
                                         v2.RandomVerticalFlip(),
                                         v2.ToImage(),
                                         v2.ToDtype(torch.float32, scale=True),])
            
            self.transform = A.Compose(
                [
                    A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
                    A.OneOf([
                        A.Blur(blur_limit=3, p=0.5),
                        A.ColorJitter(p=0.5),
                    ], p=1.0),
                ]
            )
        else:
            self.df = pd.read_csv(data_csv).sort_values('patient_id')[:10000]
            self.df =  self.df.reset_index()
            self.transform = v2.Compose([v2.ToImage(),
                                         v2.ToDtype(torch.float32, scale=True),])

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, index): 
        img_path = '/home/shivac/qml-data/'+self.df.img_path[index]
        mask_path = '/home/shivac/qml-data/'+self.df.mask_path[index] 
        img = Image.open(img_path).convert('L')
        # img = Image.open(img_path) 
        mask = Image.open(mask_path)
        if self.mode == 'train': 
            img = np.array(img) 
            mask = np.array(mask)
            data = self.transform(image=img, mask=mask)
            img = data['image'] 
            mask = data['mask'] 
            img = torch.from_numpy(img)/255.0 
            mask = torch.from_numpy(mask)/255.0
            img = rearrange(img, 'h w c -> c h w ')
            mask = mask.unsqueeze(0)
        else: 
            img, mask = self.transform(img, mask)
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
        xmin = xc - 50 
        xmax = xc + 50 
        ymin = yc - 50 
        ymax = yc + 50 
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
    dataset = Pic_to_Pic_dataset('/home/shivac/qml-data/csv_files/org_val_9.csv', mode='train')
    from torch.utils.data import DataLoader 
    loader = DataLoader(dataset, batch_size=2, shuffle=True) 
    img, mask= next(iter(loader)) 
    print(f'{img.shape, mask.shape = }')
    ToPILImage()(img[0]).save('./samples/im.png')
    ToPILImage()(mask[0]).save('./samples/crop_mask.png')
    

