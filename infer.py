import torch 
from dataset import Pic_to_Pic_dataset
from models import UNET
from torch.utils.data import DataLoader
from loss import SSIM_DICE_BCE, DiceScore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from PIL import Image 
from torchvision.transforms import ToTensor
import os 
from tqdm import tqdm 
import cv2


def imgs_to_vid(path): 
    patient_id = path.split('/')[-2]
    imgs = os.listdir(path)
    imgs = [os.path.join(path, img) for img in imgs if img.endswith('.png')]
    imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    img = cv2.imread(imgs[0])
    os.makedirs('{}/'.format(path), exist_ok=True)
    video = cv2.VideoWriter('{}/vid.mp4'.format(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (img.shape[1], img.shape[0]))

    for img_path in tqdm(imgs): 
        if not img_path.endswith('.png'): 
            continue
        img = cv2.imread(img_path)
        video.write(img)       
    video.release()
    os.system('rm ./plots/{}/*.png'.format(patient_id))


def infer(model, df):
    dice_score = DiceScore()
    patient_ids = np.unique(df.patient_id)
    assert len(patient_ids) == 1, 'Only one patient_id should be present in the dataframe'
    patient_id = patient_ids[0]
    os.makedirs('plots/{}'.format(patient_id), exist_ok=True)
    for i in tqdm(range(len(df))): 
        img_path = '/home/shivac/qml-data/' + df.loc[i].img_path
        mask_path = '/home/shivac/qml-data/' + df.loc[i].mask_path
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path) 
        mask = ToTensor()(mask).unsqueeze(0)
        img = ToTensor()(img).unsqueeze(0)
        logits = model(img.cuda())
        dice = round(dice_score(mask.cuda(), logits).item(), 2)

        plt.figure(figsize=(10, 6), facecolor='gray')
        plt.axis('off')
        plt.title('Depth: ' + str(i) + ' dice_score: ' + str(dice))
        plt.subplot(1,3,1)
        plt.title('img')
        plt.axis('off')
        plt.imshow(img[0].permute(1,2,0), cmap='gray')
        plt.subplot(1,3,2)
        plt.title('mask')
        plt.axis('off')
        plt.imshow(mask[0].permute(1,2,0), cmap='gray')
        plt.subplot(1,3,3)
        plt.title('logits')
        plt.axis('off')
        plt.imshow(logits[0].detach().cpu().permute(1,2,0), cmap='gray')
        # plt.tight_layout()
        plt.savefig('plots/{}/{}.png'.format(patient_id, i))
        plt.clf() 
        plt.close()



def main(args): 
    df = pd.read_csv('/home/shivac/qml-data/csv_files/val_10_org.csv') 
    patient_ids = np.unique(df.patient_id)
    patient_id = np.random.choice(patient_ids, 1)[0]
    df = df[df.patient_id == patient_id].sort_values('idx')
    df.reset_index(inplace=True)

    model = UNET().cuda()
    ckpt = torch.load('./ckpts/quantum_noise/56/best_unet.pth') 
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print('=====infering on patient_id: {}=====>'.format(patient_id))
    infer(model, df)
    print('=====Converting images to video=====>')
    imgs_to_vid('./plots/{}/'.format(patient_id))
    print('=====Done=====>')
    print('=====Video saved at ./plots/{}/vid.mp4=====>'.format(patient_id))

        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Infer on the model')
    parser.add_argument('--model', type=str, default='unet', help='Model to infer on')
    parser.add_argument('--ckpt', type=str, default='./ckpts/quantum_noise/56/best_unet.pth', help='Path to the model checkpoint')
    args = parser.parse_args()
    main(args)
