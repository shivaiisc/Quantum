import csv 
import torch 
import numpy as np
from dataset import Pic_to_Pic_dataset
from loss import SSIM_DICE_BCE, calc_metrics 
from utils import load_model, plot
from torch.utils.data import DataLoader
from tqdm import tqdm 
import os 
torch.set_grad_enabled(False)

def loop(model, loader, criterion, args): 
    pbar = tqdm(enumerate(loader)) 
    pbar.set_description(f'epochs:{args.curr_epoch}/{args.epochs}')
    ssim_loss_list = list() 
    dice_loss_list = list() 
    bce_loss_list = list() 
    total_loss_list = list()
    acc_list = list()
    precision_list = list() 
    recall_list = list() 
    f1_score_list = list() 

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

        metrics = calc_metrics(logits, y)
        loss_dict = {'dice_loss': round(dice_loss.item(), 4),
                    'ssim_loss': round(ssim_loss.item(), 4),
                    'bce_loss': round(bce_loss.item(), 4),
                    'total_loss': round(loss.item(), 4)}
        ssim_loss_list.append(loss_dict['ssim_loss'])
        dice_loss_list.append(loss_dict['dice_loss'])
        bce_loss_list.append(loss_dict['bce_loss'])
        total_loss_list.append(loss_dict['total_loss'])
        acc_list.append(metrics['acc'])
        precision_list.append(metrics['precision'])
        recall_list.append(metrics['recall']) 
        f1_score_list.append(metrics['f1_score']) 
        pbar.set_postfix({**metrics, **loss_dict}, refresh=idx%10==0)
    
    loss_dct = {'ssim_loss': round(np.mean(ssim_loss_list), 4),
                'dice_loss': round(np.mean(dice_loss_list), 4),
                'bce_loss': round(np.mean(dice_loss_list), 4),
                'total_loss': round(np.mean(total_loss_list), 4),
                'f1_score': round(np.mean(f1_score_list), 4),
                'precision': round(np.mean(precision_list, 4)),
                'recall': round(np.mean(recall_list), 4),
                'acc': round(np.mean(acc_list), 4)}
    
    return loss_dct
        
def main(args):
    test_data = Pic_to_Pic_dataset(args.data_csv)
    loader = DataLoader(test_data, args.batch_size) 

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

    args.load_path = f'./ckpts/{args.model_name}.pth'
    args.load_best_path = f'./ckpts/best_{args.model_name}.pth'
    model = load_model(model, args, best=args.best)
    
    criterion = SSIM_DICE_BCE() 
    if args.parallel:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    
    if not os.path.exists('./res/{}'.format(args.model_name)):
        os.makedirs('./res/{}'.format(args.model_name))
    f = open(f'./res/{args.model_name}/{args.model_name}_test.csv', 'w')
    row = ['dice_loss', 'ssim_loss', 'bce_loss', 'total_loss',
           'acc', 'precision', 'recall', 'f1_score']

    dct = {k:k for k in row}
    args.log = csv.DictWriter(f, dct.keys()) 
    args.log.writerow(dct)
    print(args)
    dct = loop(model, loader, criterion, args)
    args.log.writerow(dct)
    f.close() 

    

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-m', '--model_name', type=str, default='unet')
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-es', '--early_stop', type=int, default=10) 
    parser.add_argument('-dp', '--data_csv', type=str, default='../../qml-data/qml_mns.csv') 
    parser.add_argument('-ic', '--in_ch', type=int, default=1)
    parser.add_argument('-oc', '--out_ch', type=int, default=1)
    parser.add_argument('-nq', '--n_qubits', type=int, default=28)
    parser.add_argument('-sr', '--ssim_ratio', type=float, default=0.33)
    parser.add_argument('-dr', '--dice_ratio', type=float, default=0.33)
    parser.add_argument('-br', '--bce_ratio', type=float, default=0.33)
    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-b', '--best', type=int, default=1)
    args = parser.parse_args()

    args.device = 'cuda'
    args.early_stopping_idx = 0 

    main(args)

