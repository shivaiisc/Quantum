import csv 
import torch 
import numpy as np
from dataset import Pic_to_Pic_dataset
from loss import SSIM_DICE_BCE, calc_metrics 
from utils import save_model, load_model, plot
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam 
from torchvision.utils import make_grid 
from tqdm import tqdm 
from einops import rearrange
import os 

def loop(model, loader, optimizer, criterion, args, mode='train'): 
    pbar = tqdm(enumerate(loader)) 
    pbar.set_description(f'epochs:{args.curr_epoch}/{args.epochs}')
    ssim_loss_list = list() 
    dice_loss_list = list() 
    bce_loss_list = list() 
    total_loss_list = list()
    for idx, (x, y) in pbar: 
        x = x.to(args.device)
        y = y.to(args.device)
        logits = model(x) 
        loss = criterion(logits, y)
        
        ssim_loss = loss['ssim_loss'] 
        dice_loss = loss['dice_loss'] 
        bce_loss = loss['bce_loss'] 
        loss = args.ssim_ratio * ssim_loss + \
            args.dice_ratio * dice_loss + \
            args.bce_ratio * bce_loss 

        if mode == 'train': 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
        # metrics = calc_metrics(logits, y)
        log_dict = {'dice_loss': round(dice_loss.item(), 4),
                    'ssim_loss': round(ssim_loss.item(), 4),
                    'bce_loss': round(bce_loss.item(), 4),
                    'total_loss': round(loss.item(), 4),
                    'mode': mode,
                    'es': f'{args.early_stopping_idx}/{args.early_stop}'}
        ssim_loss_list.append(log_dict['ssim_loss'])
        dice_loss_list.append(log_dict['dice_loss'])
        bce_loss_list.append(log_dict['bce_loss'])
        total_loss_list.append(log_dict['total_loss'])
        # pbar.set_postfix({**metrics, **log_dict})
        pbar.set_postfix(log_dict, refresh=idx%10==0)
    loss_dct = {f'{mode}_ssim_loss': round(np.mean(ssim_loss_list), 4),
                f'{mode}_dice_loss': round(np.mean(dice_loss_list), 4),
                f'{mode}_bce_loss': round(np.mean(dice_loss_list), 4),
                f'{mode}_total_loss': round(np.mean(total_loss_list), 4)}
    
    return loss_dct
        

def train(model, loaders, optimizer, criterion, args):
    train_loader = loaders['train']
    test_loader = loaders['test']
    val_loader = None 
    val_loss = None
    val_loader = loaders['val']

    best_loss = float('inf') 
    for epoch in range(args.epochs): 
        args.curr_epoch = epoch
        
        model.train() 
        train_loss = loop(model, train_loader, optimizer, 
             criterion, args, mode='train')
        print('Training loss', train_loss)
    
        with torch.no_grad(): 
            model.eval()
            val_loss = loop(model, val_loader, optimizer,
                    criterion, args, mode='val') 
        print('val loss', val_loss)
        loss = val_loss['val_total_loss'] 
        log_dct = {**train_loss, **val_loss}
        args.log.writerow(log_dct)
        
        if loss < best_loss: 
            best_loss = loss 
            args.early_stopping_idx = 0 
            save_model(model, loss, args, best=True)
        elif args.early_stopping_idx > args.early_stop: 
            print('-'*10, 'Earyly stopping', '-'*10)
            break
        else:
            args.early_stopping_idx += 1

        save_model(model, loss, args, best=False)
    print('==========Training done==============') 


    with torch.no_grad(): 
        test_loss = loop(model, test_loader, optimizer,\
                         criterion, args, mode='test') 
        print('Test loss', test_loss) 

def main(args):
    dataset = Pic_to_Pic_dataset(args.data_csv)
    train_data, val_data, test_data = random_split(dataset, [0.8, 0.1, 0.1]) 
    loaders = {'train': DataLoader(train_data, args.batch_size),
               'val': DataLoader(val_data, args.batch_size),
               'test': DataLoader(test_data, args.batch_size)} 
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
    if args.parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    args.save_path = f'./ckpts/{args.model_name}.pth'
    args.save_best_path = f'./ckpts/best_{args.model_name}.pth'
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = SSIM_DICE_BCE() 
    
    f = open('plot.csv', 'w')
    row = ['train_dice_loss', 'train_ssim_loss', \
    'train_bce_loss', 'train_total_loss', 'val_dice_loss',\
    'val_ssim_loss', 'val_bce_loss', 'val_total_loss']

    dct = {k:k for k in row}
    args.log = csv.DictWriter(f, dct.keys()) 
    args.log.writerow(dct)
    print(args)
    train(model, loaders, optimizer, criterion, args)
    f.close() 

    plot('./plot.csv')
    os.system('~/git_img.sh')
    

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-m', '--model_name', type=str, default='unet')
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-es', '--early_stop', type=int, default=10) 
    parser.add_argument('-dp', '--data_csv', type=str, default='../../qml-data/qml_mns.csv') 
    parser.add_argument('-ic', '--in_ch', type=int, default=1)
    parser.add_argument('-oc', '--out_ch', type=int, default=1)
    parser.add_argument('-nq', '--n_qubits', type=int, default=28)
    parser.add_argument('-sr', '--ssim_ratio', type=float, default=0.33)
    parser.add_argument('-dr', '--dice_ratio', type=float, default=0.33)
    parser.add_argument('-br', '--bce_ratio', type=float, default=0.33)
    parser.add_argument('-p', '--parallel', type=int, default=1)
    args = parser.parse_args()

    args.device = 'cuda'
    args.early_stopping_idx = 0 

    main(args)

