import csv 
import torch 
import numpy as np
from dataset import Cond_Pic_to_Pic_dataset
from loss import SSIM_DICE_BCE, calc_metrics 
from utils import save_model, load_model, plot, get_model
from torch.optim import Adam 
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import os 

import os

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1" 




def loop(model, loader, optimizer, criterion, args, mode='train'): 
    pbar = tqdm(enumerate(loader)) 
    pbar.set_description(f'epochs:{args.curr_epoch}/{args.epochs}')
    ssim_loss_list = list() 
    dice_loss_list = list() 
    bce_loss_list = list() 
    total_loss_list = list()
    for idx, (x, y, idxs) in pbar: 
        x = x.to(args.device)
        y = y.to(args.device)
        idxs = idxs.to(args.device)
        idxs = torch.where(idxs <290, idxs, 290)
        cond = torch.zeros_like(x, device=args.device).repeat(1, 290, 1, 1)
        cond[:, idxs] = torch.ones(x.shape[-2:], device=args.device, dtype=torch.float32)
        x = torch.cat([x, cond], dim=1)
        logits = model(x) 
        loss = criterion(logits, y)
        # print(loss)
        
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
        f = open(args.csv_path, 'a')
        args.log = csv.DictWriter(f, log_dct.keys()) 
        args.log.writerow(log_dct)
        f.close() 
        plot(args.csv_path, args.plot_path)
        os.system('../g.sh>>del.txt')
    print('==========Training done==============') 


    with torch.no_grad(): 
        test_loss = loop(model, test_loader, optimizer,\
                         criterion, args, mode='test') 
        print('Test loss', test_loss) 

def main(args):
    train_data = Cond_Pic_to_Pic_dataset(args.train_csv)
    val_data = Cond_Pic_to_Pic_dataset(args.val_csv)
    test_data = Cond_Pic_to_Pic_dataset(args.test_csv)
    loaders = {'train': DataLoader(train_data, args.batch_size),
               'val': DataLoader(val_data, args.batch_size),
               'test': DataLoader(test_data, args.batch_size)} 
    model = get_model(args)
    if args.parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    args.save_path = f'./ckpts/{args.model_name}.pth'
    args.save_best_path = f'./ckpts/best_{args.model_name}.pth'
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = SSIM_DICE_BCE() 
    dir = str(len(os.listdir('./logs/cond/')))
    os.makedirs('./logs/cond/{}'.format(dir))
    args.csv_path = './logs/cond/' + dir + '/log.csv'
    args.plot_path = './logs/cond/' + dir + '/train_plot.png'
    config_path = './logs/cond/' + dir + '/config.txt'
    args.config_file = open(config_path, 'w')
    row = ['train_dice_loss', 'train_ssim_loss', \
    'train_bce_loss', 'train_total_loss', 'val_dice_loss',\
    'val_ssim_loss', 'val_bce_loss', 'val_total_loss']

    dct = {k:k for k in row}
    f = open(args.csv_path, 'w')
    args.log = csv.DictWriter(f, dct.keys()) 
    args.log.writerow(dct)
    f.close() 
    print(args)
    args.config_file.write(str(vars(args)))
    train(model, loaders, optimizer, criterion, args)
    plot(args.csv_path, args.plot_path)
    os.system('../g.sh')

    

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-m', '--model_name', type=str, default='unet')
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-es', '--early_stop', type=int, default=10) 
    parser.add_argument('--train_csv', type=str, default='../../../qml-data/csv_files/train_80.csv') 
    parser.add_argument('--val_csv', type=str, default='../../../qml-data/csv_files/val_10.csv') 
    parser.add_argument('--test_csv', type=str, default='../../../qml-data/csv_files/test_20.csv') 
    parser.add_argument('-ic', '--in_ch', type=int, default=291)
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

