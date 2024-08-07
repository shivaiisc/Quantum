import torch 
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image 
from utils import pth_to_depth_vid, feat_to_vid
import os 
from einops import rearrange
from tqdm import tqdm 

def forward_hook(inst, ip, op): 
    # op = (op-torch.min(op))/(torch.max(op)-torch.min(op)) 
    idx = os.listdir(args.vis_dir)
    print(op.shape)
    exit()
    ToPILImage()(op).save('op_{}.png'.format(idx))
    return 
    cos_sim = rearrange(op, '1 c h w -> (h w) c')
    norm = torch.linalg.norm(cos_sim, dim=1).view(-1, 1)
    norm = norm @ norm.t()
    cos_sim = cos_sim @ cos_sim.t() 
    cos_sim = cos_sim/norm
    idx = next(args.gen)#len(os.listdir(args.vis_path))
    ToPILImage()(cos_sim.unsqueeze(0)).save(args.mp4_path+f'_{idx}_cos_sim.png')
    return
    op = rearrange(op, '1 c h w -> c 1 h w')
    op = op.repeat(1, 3, 1, 1)
    pth_to_depth_vid(op.cpu(), args.img_file, os.path.join(args.mp4_path + '.mp4'))
    # pth_to_vid(op.cpu(), os.path.join(args.mp4_path + '.mp4'))
    
def gener(lst):
    for ins in lst:
        yield ins



def main(args): 
    if args.model_name == 'q_unet': 
        from models import Q_UNET 
        model = Q_UNET(n_qubits=args.n_qubits).to(args.dev)
        model.load_state_dict(torch.load(args.model_path)['model_state'])
        model.down3.register_forward_hook(forward_hook)
        model.down4.register_forward_hook(forward_hook)
        model.qml_encoder.register_forward_hook(forward_hook)
        model.qml_decoder.register_forward_hook(forward_hook)
        model.up1.register_forward_hook(forward_hook)
        # model.up2.register_forward_hook(forward_hook)
        lst = ['down3', 'down4', 'qml_encoder', 'qml_decoder', 'up1']
    elif args.model_name == 'unet': 
        from models import UNET 
        model = UNET().to(args.dev)
        model.load_state_dict(torch.load(args.model_path)['model_state'])
        # model.down2.register_forward_hook(forward_hook)
        model.down3.register_forward_hook(forward_hook)
        model.down4.register_forward_hook(forward_hook)
        model.up1.register_forward_hook(forward_hook)
        # model.up2.register_forward_hook(forward_hook)
        lst = ['down3', 'down4', 'up1']
    else:
        raise ValueError('Model Unavailable')

    if not os.path.exists(args.vis_dir):
        os.mkdir(args.vis_dir)
    imgs = os.listdir(args.img_dir)
    for img_file in tqdm(imgs):
        args.gen = gener(lst)
        img_path = os.path.join(args.img_dir, img_file) 
        img = Image.open(img_path).convert('L') 
        img = ToTensor()(img).to(args.dev).unsqueeze(0)
        args.vis_path = os.path.join(args.vis_dir, img_file[:-4])
        if not os.path.exists(args.vis_path):
            os.mkdir(args.vis_path)
        args.mp4_path = os.path.join(args.vis_path, img_file[:-4])
        args.img_file = img_file[:-4]
        _ = model(img)

    dct = {ins: [] for ins in lst}
    for depth in tqdm(os.listdir(args.vis_dir)):
        depth = os.path.join(args.vis_dir, depth)
        for img_path in os.listdir(depth):
            for ins in lst:
                if ins in img_path: 
                    dct[ins].append(os.path.join(depth, img_path))
    for ins in lst:
        mp4_path = '/'.join(dct[ins][0].split('/')[:-2])
        mp4_path += f'/{ins}_feat'
        feat_to_vid(sorted(dct[ins], key=lambda k: int(k.split('/')[-2])),
                    mp4_path, 1)
    
    
    


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
