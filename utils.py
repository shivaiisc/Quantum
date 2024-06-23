import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm 
from einops import rearrange
import torch 
import os 
import numpy as np
from PIL import Image 
import cv2 
from torchvision.transforms import ToTensor 


def get_model(args): 
    if args.model_name == 'unet':
        from models import UNET
        model = UNET(in_ch=args.in_ch, out_ch=args.out_ch).to(args.device)
    elif args.model_name == 'u2net': 
        from models import U2NET
        model = U2NET(in_ch=args.in_ch, out_ch=args.out_ch).to(args.device) 
    elif args.model_name == 'q_unet': 
        from models import Q_UNET 
        model = Q_UNET(in_ch=args.in_ch, out_ch=args.out_ch, n_qubits=args.n_qubits).to(args.device)
    else:
        print('model Unavailable')
        exit()
    return model


def imgs_to_vid(pre_imgs, out_path, mask=False):
    path = '/home/shivac/qml-data/'
    out_path = f'./vids/{out_path}/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if mask:
        out_video_name = 'mak.mp4'
    else:
        out_video_name = 'im.mp4'
    out_video_full_path = out_path+out_video_name
    print(out_video_full_path)

    img = []

    for i in pre_imgs:
        i = path+i
        img.append(i)

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame = cv2.imread(img[0])
    size = list(frame.shape)
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(out_video_full_path, cv2_fourcc, 60, size) #output video name, fourcc, fps, size

    for i in range(len(img)): 
        im = Image.open(img[i])#.convert('L')
        im = ToTensor()(im).permute(1, 2, 0)
        im = im * 255.0
        im = im.numpy().astype(np.uint8)
        im_cv2 = cv2.imread(img[i])#[:, :, :1]
        if mask: 
            im_cv2 = np.where(im_cv2 == 255, 0, 255)
            im_cv2 = im_cv2.astype(np.uint8)
        break
        video.write(im_cv2)
    video.release()

def imgs_to_pth(img_path, mask_path, src_path='./.data'):
    idx = len(os.listdir(src_path)) 
    os.makedirs(os.path.join(src_path, str(idx)))
    # all_img = list() 
    all_mask = list() 
    for im_path, mk_path in zip(img_path, mask_path): 
        # img = ToTensor()(Image.open('../../qml-data/' + im_path).convert('L'))
        mask = ToTensor()(Image.open('../../qml-data/' + mk_path))
        mask = mask.to(torch.long)

        # mask = torch.where(mask<=0.5, 1.0, 0.0)
        # all_img.append(img)
        all_mask.append(mask) 
    # all_img = torch.stack(all_img)
    all_mask = torch.stack(all_mask)
    torch.save(all_mask[:290], os.path.join(src_path, str(idx), 'mask.pth'))
    Image.open('../../qml-data/'+img_path[0]).convert('L').save(\
                    os.path.join(src_path, str(idx), 'img.png'))
    # torch.save(all_img, os.path.join(src_path, str(idx), 'img.pth'))

def pth_to_vid(pth, path='./res/vid.mp4'): 

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pth = pth * 255.0
    pth = pth.to(torch.long)
    size = list(pth.shape)
    del size[0]
    del size[0]
    size.reverse()

    video = cv2.VideoWriter(path, cv2_fourcc, 60, size) #output video name, fourcc, fps, size
    pth = rearrange(pth, 'b c h w -> b h w c')

    for img in pth: 
        video.write(img.numpy().astype(np.uint8))
    video.release()

def save_model(model, loss, args, best=False): 
    dct = {'model_state':model.module.state_dict() if args.parallel \
                            else model.state_dict()}
    print('====saving model====')
    torch.save(dct, args.save_best_path if best\
               else args.save_path)


def load_model(model, args, best): 
    dct = torch.load(args.load_best_path if best\
                     else args.load_path) 
    print('====loading model====')
    args.curr_epoch = dct['curr_epoch']
    model.load_state_dict(dct['model_state'])
    print('config:', dct['config'])
    return model

def plot(csv_path, save_path):
    df = pd.read_csv(csv_path)
    train_data = sorted(df.columns)[:len(df.columns)//2]
    val_data = sorted(df.columns)[len(df.columns)//2:]
    # plt.figure(figsize=(20, 16))
    for idx, (train, val) in enumerate(zip(train_data, val_data)):
        plt.plot(df[train], label='train')
        plt.plot(df[val], label='val')

        plt.title(val[4:])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend() 
        plt.grid()
        plt.savefig(save_path + f'{val[4:]}.png')
        plt.clf()

def main(args): 
    df = pd.read_csv(args.path).sort_values(by='idx') 
    unique_patients = np.unique(df.patient_id)
    print('size of data:', len(df))
    print('num of patients', len(unique_patients))
    print('data columsn', df.columns)
    df = df.groupby('patient_id') 
    for idx, patient in tqdm(enumerate(unique_patients), total=100): 
        patient_df = df.get_group(patient)
        l = len(patient_df) 
        if l < 290: 
            continue 
        if idx > 100:
            break
        imgs_to_vid(list(patient_df.img_path), patient)
        imgs_to_vid(list(patient_df.mask_path), patient, mask=True)
        # imgs_to_pth(list(patient_df.img_path), list(patient_df.mask_path))
        exit()

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-p', '--path', type=str, default='/home/shivac/qml-data/csv_files/org_99.csv')
    args = parser.parse_args() 

    main(args)

