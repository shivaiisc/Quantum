import torch 
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image 
from utils import pth_to_depth_vid, pth_to_vid
import os 
from einops import rearrange
from tqdm import tqdm 

def forward_hook(inst, ip, op): 
    # op = (op-torch.min(op))/(torch.max(op)-torch.min(op)) 
    cos_sim = rearrange(op, '1 c h w -> (h w) c')
    norm = torch.linalg.norm(cos_sim, dim=1).view(-1, 1)
    norm = norm @ norm.t()
    cos_sim = cos_sim @ cos_sim.t() 
    cos_sim = cos_sim/norm
    idx = len(os.listdir(args.vis_dir))
    ToPILImage()(cos_sim.unsqueeze(0)).save(args.mp4_path+f'_{idx}_cos_sim.png')
    return
    op = rearrange(op, '1 c h w -> c 1 h w')
    op = op.repeat(1, 3, 1, 1)
    pth_to_depth_vid(op.cpu(), args.img_file, os.path.join(args.mp4_path + '.mp4'))
    # pth_to_vid(op.cpu(), os.path.join(args.mp4_path + '.mp4'))
    



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
        # model.down2.register_forward_hook(forward_hook)
        # model.down3.register_forward_hook(forward_hook)
        model.down4.register_forward_hook(forward_hook)

    else:
        raise ValueError('Model Unavailable')
    if not os.path.exists(args.vis_dir):
        os.mkdir(args.vis_dir)
    imgs = os.listdir(args.img_dir)
    for img_file in tqdm(imgs):
        img_path = os.path.join(args.img_dir, img_file) 
        img = Image.open(img_path).convert('L') 
        img = ToTensor()(img).to(args.dev).unsqueeze(0)
        args.mp4_path = os.path.join(args.vis_dir, img_file[:-4])
        args.img_file = img_file[:-4]
        logits = model(img)

        


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
