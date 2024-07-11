import csv 
from curtsies.fmtfuncs import green, red, blue 
import pickle
import torch 
import numpy as np
from dataset import Pic_to_Pic_dataset, Crop_dataset
from loss import SSIM_DICE_BCE, calc_metrics 
from utils import save_model, load_model, plot, get_model
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam 
from torchvision.utils import make_grid 
from torchvision import transforms as T
from tqdm import tqdm 
from einops import rearrange
import os 


def loop(model, loader, optimizer, criterion, args, mode='train'): 
    pbar = tqdm(enumerate(loader), total=len(loader)) 
    pbar.set_description(f'epochs:{args.curr_epoch}/{args.epochs}')
    ssim_loss_list = list() 
    dice_loss_list = list() 
    bce_loss_list = list() 
    total_loss_list = list()
    for idx, (x, y) in pbar: 
        x = x.to(args.device)
        y = y.to(args.device)
        if mode == 'train':
            x = x + torch.randn_like(x, device=args.device)*args.noise
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
        
        log_dict.pop('ssim_loss') 
        log_dict.pop('bce_loss')
        pbar.set_postfix(log_dict, refresh=idx%10==0)
    loss_dct = {f'{mode}_ssim_loss': round(np.mean(ssim_loss_list), 4),
                f'{mode}_dice_loss': round(np.mean(dice_loss_list), 4),
                f'{mode}_bce_loss': round(np.mean(bce_loss_list), 4),
                f'{mode}_total_loss': round(np.mean(total_loss_list), 4)}
    
    return loss_dct
        

def train(model , loaders, optimizer, criterion, args):
    train_loader = loaders['train']
    test_loader = loaders['test']
    val_loader = loaders['val']

    
    best_loss = float('inf') 
    if args.from_scratch: 
        ckpt = torch.load(args.save_path) 
        model_state = ckpt['model_state'] 
        curr_epoch = ckpt[ 'epochs']
        model.load_state_dict(model_state)
        epochs_range = range(curr_epoch, args.epochs)
    else: 
        epochs_range = range(args.epochs) 
        
    for epoch in epochs_range: 
        args.curr_epoch = epoch
        
        model.train() 
        train_loss = loop(model, train_loader, optimizer, 
             criterion, args, mode='train')
        print(blue(f'Training loss {train_loss}'))
    
        model.eval()
        with torch.no_grad(): 
            val_loss = loop(model, val_loader, optimizer,
                    criterion, args, mode='val') 
        loss = val_loss['val_total_loss'] 
        dct = {**train_loss, **val_loss}
        
        if loss < best_loss: 
            best_loss = loss 
            args.early_stopping_idx = 0 
            save_model(model, loss, args, best=True)
            print(green(f'val loss {val_loss}'))
        elif args.early_stopping_idx >= args.early_stop: 
            args.early_stopping_idx -= 1
            print('-'*10, 'Earyly stopping', '-'*10)
            break
        else:
            print(red(f'val loss {val_loss}'))
            args.early_stopping_idx += 1

        save_model(model, loss, args, best=False)

        f = open(args.csv_path, 'a')
        args.log = csv.DictWriter(f, dct.keys()) 
        args.log.writerow(dct)
        f.close()
        plot(args.csv_path, args.plot_path)
        os.system('./g.sh&>>del.txt')

    print('==========Training done==============') 


    model.eval()
    with torch.no_grad(): 
        test_loss = loop(model, test_loader, optimizer,\
                         criterion, args, mode='test') 
        print(blue(f'Test loss: {test_loss}'))  

    with open(args.logs_path + '/test_results.txt', 'w') as f:
        f.write(f'test loss: {test_loss}')
    os.system('./g.sh&>>del.txt')


def main(args):
    train_data = Pic_to_Pic_dataset(args.train_csv, mode='train')
    val_data = Pic_to_Pic_dataset(args.val_csv, mode='val')
    test_data = Pic_to_Pic_dataset(args.test_csv, mode='test')

    loaders = {'train': DataLoader(train_data, args.batch_size, shuffle=True),
               'val': DataLoader(val_data, args.batch_size),
               'test': DataLoader(test_data, args.batch_size)} 
 
    print(f'{len(train_data) = }')
    print(f'{len(val_data) = }')
    print(f'{len(test_data) = }')
    model = get_model(args)
    if args.parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    
    
    ckpt_dir = f'./ckpts/{args.experiment}/'
    os.makedirs(ckpt_dir, exist_ok=True)
    args.save_path = f'{ckpt_dir}/{args.model_name}_last.pth'
    args.save_best_path = f'{ckpt_dir}/best_{args.model_name}.pth'
    args.logs_path = f'./logs/{args.experiment}/'
    os.makedirs(args.logs_path)
    with open(args.logs_path+'/config.pkl', 'wb') as f:
        pickle.dump(vars(args), f)
    args.csv_path = f'{args.logs_path}/log.csv'
    args.plot_path = f'{args.logs_path}/plots/'
    if not os.path.exists(args.plot_path):
        os.mkdir(args.plot_path)
    config_txt_path = f'{args.logs_path}/config.txt'
    
    optimizer = Adam(params = filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr)
    # optimizer = SGD(params = filter(lambda p: p.requires_grad, model.parameters()),
    #                  lr=args.lr)
    criterion = SSIM_DICE_BCE() 
    
    row = ['train_ssim_loss', 'train_dice_loss', \
    'train_bce_loss', 'train_total_loss', 'val_ssim_loss',\
    'val_dice_loss', 'val_bce_loss', 'val_total_loss']

    dct = {k:k for k in row}
    f = open(args.csv_path, 'w')
    args.log = csv.DictWriter(f, dct.keys()) 
    args.log.writerow(dct)
    f.close()
    print(vars(args))
    with open(config_txt_path, 'w') as f: 
        f.write(str(vars(args)))
    train(model, loaders, optimizer, criterion, args)

    

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-ep', '--epochs', type=int, default=50)
    parser.add_argument('-rs', '--random_split', type=int, default=1)
    parser.add_argument('-n', '--noise', type=float, default=0.0)
    parser.add_argument('-exp', '--experiment', type=str, default='quantum_noise')
    parser.add_argument('-m', '--model_name', type=str, default='unet')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-es', '--early_stop', type=int, default=6) 
    parser.add_argument('-trc', '--train_csv', type=str, default='../../qml-data/csv_files/train_75_org.csv') 
    parser.add_argument('-vc', '--val_csv', type=str, default='../../qml-data/csv_files/val_10_org.csv') 
    parser.add_argument('-tc', '--test_csv', type=str, default='../../qml-data/csv_files/test_14_org.csv') 
    parser.add_argument('-ic', '--in_ch', type=int, default=1)
    parser.add_argument('-oc', '--out_ch', type=int, default=1)
    parser.add_argument('-nq', '--n_qubits', type=int, default=28)
    parser.add_argument('-sr', '--ssim_ratio', type=float, default=0.33)
    parser.add_argument('-dr', '--dice_ratio', type=float, default=0.33)
    parser.add_argument('-br', '--bce_ratio', type=float, default=0.33)
    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-th', '--threshold', type=float, default=0.7)
    parser.add_argument('-dev', '--device', type=str, default='cuda:0')
    parser.add_argument('-fs', '--from_scratch', type=int, default=1) 

    args = parser.parse_args()

    # print(args)
    args.early_stopping_idx = 0 

    main(args)
