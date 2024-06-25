from re import U
import torch 
from torchvision.transforms import ToTensor 
from PIL import Image 
import os 

def main(args): 
    if args.model_name == 'q_unet': 
        from models import Q_UNET 
        model = Q_UNET(n_qubits=args.n_qubits).to(args.dev)
    elif args.model_name == 'unet': 
        from models import UNET 
        model = UNET().to(args.dev)
    else:
        raise ValueError('Model Unavailable')
    imgs = os.listdir(args.img_dir)
    for img_file in imgs:
        img_path = os.path.join(args.img_dir, img_file) 
        img = Image.open(img_path).convert('L') 
        img = ToTensor()(img)
        logits = model(img)
        print(logits.shape)

        


if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model_name', type=str) 
    parser.add_argument('--n_qubits', type=int, default=28) 
    parser.add_argument('--img_dir', type=str) 
    parser.add_argument('--vis_dir', type=str)

    args = parser.parse_args() 
    args.dev = 'cuda:1'
    print(vars(args))
    main(args)
