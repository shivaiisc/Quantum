import torch 
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image 
from utils import pth_to_vid
import os 
from einops import rearrange

def forward_hook(inst, ip, op): 
    print('Here') 
    op = (op-torch.min(op))/(torch.max(op)-torch.min(op)) 
    op = rearrange(op, '1 c h w -> c 1 h w')
    op = op.repeat(1, 3, 1, 1)
    print(op.shape, torch.max(op), torch.min(op))
    pth_to_vid(op.cpu(), os.path.join(args.vis_dir + args.model_name+'.mp4'))



def main(args): 
    if args.model_name == 'q_unet': 
        from models import Q_UNET 
        model = Q_UNET(n_qubits=args.n_qubits).to(args.dev)
        model.load_state_dict(torch.load(args.model_path)['model_state'])
        model.qml_decoder.register_forward_hook(forward_hook)
    elif args.model_name == 'unet': 
        from models import UNET 
        model = UNET().to(args.dev)
        model.load_state_dict(torch.load(args.model_path)['model_state'])
        model.down4.register_forward_hook(forward_hook)

    else:
        raise ValueError('Model Unavailable')
    imgs = os.listdir(args.img_dir)
    for img_file in imgs:
        img_path = os.path.join(args.img_dir, img_file) 
        img = Image.open(img_path).convert('L') 
        img = ToTensor()(img).to(args.dev).unsqueeze(0)
        logits = model(img)
        print(logits.shape)
        exit()

        


if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model_name', type=str) 
    parser.add_argument('--model_path', type=str) 
    parser.add_argument('--n_qubits', type=int, default=28) 
    parser.add_argument('--img_dir', type=str) 
    parser.add_argument('--vis_dir', type=str)

    args = parser.parse_args() 
    args.dev = 'cuda:1'
    print(vars(args))
    main(args)
