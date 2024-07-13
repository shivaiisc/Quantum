import os 
from PIL import ImageDraw
from torchvision.utils import make_grid
from PIL import Image
from typing import List 
from torch import cos, cosine_similarity, nn
from os.path import isdir, join
from os import makedirs, removedirs
import torch 
from torch.nn import functional as F
import pennylane as qml 
from einops import rearrange
from torchvision.transforms import ToPILImage, Resize, ToTensor


class Q_Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int=0, num_layers: int=0, num_qubits: int=0):
        super().__init__()
        if num_qubits == 0:
            num_qubits = kernel_size**2 * in_channels
        if num_layers == 0:
            num_layers = kernel_size**2
        assert num_qubits == kernel_size**2 * in_channels, "The kernel size must be a square of the number of qubits"
        dev = qml.device("default.qubit", wires=num_qubits)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=list(range(num_qubits))[-1]))]

        weight_shapes = {"weights": (num_layers, num_qubits)}

        self.qlayer_list = nn.ModuleList([qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(self.out_channels)])

    def forward(self, x):
        assert len(x.shape) == 4, "The input tensor must be 4D"
        assert x.shape[1] == self.in_channels, "The number of input channels must be equal to the in_channels"
        res = list()
        x = x.unfold(2, self.kernel_size, self.stride)
        x = x.unfold(3, self.kernel_size, self.stride)
        x = rearrange(x, 'b c h w i j -> b h w (c i j)')    
        bs, h, w, _ = x.shape
        for i in range(self.out_channels):
            res.append(self.qlayer_list[i](x).view(bs, h, w))        
        x = torch.stack(res, dim=1)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__() 
        mid_ch = out_ch if not mid_ch else mid_ch
        self.net = nn.Sequential(nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(mid_ch),
                                 # nn.ReLU(inplace=True),
                                 nn.PReLU(num_parameters=mid_ch),

                                 nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.PReLU(num_parameters=out_ch))
                                 # nn.ReLU(inplace=True))


    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__() 
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        ypad = x2.shape[2] - x1.shape[2]
        xpad = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [xpad//2, xpad-xpad//2, ypad//2, ypad-ypad//2]) 

        x = torch.cat([x2, x1], dim =1)
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__() 
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                 DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)

class UNET(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, bilinear=True):
        super().__init__() 
        self.ch = DoubleConv(in_ch, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024//factor)

        self.up1 = UpBlock(1024, 512//factor, bilinear)
        self.up2 = UpBlock(512, 256//factor, bilinear)
        self.up3 = UpBlock(256, 128//factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.out = nn.Sequential(nn.Conv2d(64, out_ch, kernel_size=1, stride=1),
                                 nn.Sigmoid())

    def forward(self, x, save_path):
        makedirs(save_path, exist_ok=True)
        res_path = join(save_path, 'ip.pt')
        torch.save(x, res_path)

        res_path = join(save_path, 'ch.pt')
        x0 = self.ch(x)
        torch.save(x0, res_path)

        res_path = join(save_path, 'down1.pt')
        x1 = self.down1(x0)
        torch.save(x1, res_path)

        res_path = join(save_path, 'down2.pt')
        x2 = self.down2(x1)
        torch.save(x2, res_path)

        res_path = join(save_path, 'down3.pt')
        x3 = self.down3(x2)
        torch.save(x3, res_path)

        res_path = join(save_path, 'down4.pt')
        x = self.down4(x3)
        torch.save(x, res_path)

        res_path = join(save_path, 'up1.pt')
        x = self.up1(x, x3)
        torch.save(x, res_path)

        res_path = join(save_path, 'up2.pt')
        x = self.up2(x, x2)
        torch.save(x, res_path)

        res_path = join(save_path, 'up3.pt')
        x = self.up3(x, x1)
        torch.save(x, res_path)

        res_path = join(save_path, 'up4.pt')
        x = self.up4(x, x0)
        torch.save(x, res_path)

        res_path = join(save_path, 'out.pt')
        out = self.out(x)
        torch.save(out, res_path)

        return out 









class Q_UNET(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_qubits=28, bilinear=True):
        super().__init__() 
        self.n_qubits = n_qubits
        self.ch = DoubleConv(in_ch, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.qml_encoder = nn.Sequential(nn.Conv2d(512, 1, 1, 1), 
                                         Q_Conv2d(1, 1, 2, 2, num_layers=2),
                                         nn.Conv2d(1, 512, 1, 1))
                                       
        # self.qml_encoder = nn.Conv2d(512, 1, 1, 1)
        # self.qml_lay = Quanv(n_qubits, n_qubits)
        # self.qml_decoder = nn.Conv2d(1, 512, 1, 1)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024//factor)

        self.up1 = UpBlock(1024, 512//factor, bilinear)
        self.up2 = UpBlock(512, 256//factor, bilinear)
        self.up3 = UpBlock(256, 128//factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.out = nn.Sequential(nn.Conv2d(64, out_ch, kernel_size=1, stride=1),
                                 nn.Sigmoid())

    def forward(self, x, save_path):
        makedirs(save_path, exist_ok=True)
        res_path = join(save_path, 'ip.pt')
        torch.save(x, res_path)

        res_path = join(save_path, 'ch.pt')
        x0 = self.ch(x)
        torch.save(x0, res_path)

        res_path = join(save_path, 'down1.pt')
        x1 = self.down1(x0)
        torch.save(x1, res_path)

        res_path = join(save_path, 'down2.pt')
        x2 = self.down2(x1)
        torch.save(x2, res_path)

        res_path = join(save_path, 'down3.pt')
        x3 = self.down3(x2)
        torch.save(x3, res_path)

        res_path = join(save_path, 'down4.pt')
        x = self.down4(x3)
        torch.save(x, res_path)

        res_path = join(save_path, 'qml_encoder.pt')
        x = self.qml_encoder(x)
        torch.save(x, res_path)

        res_path = join(save_path, 'up1.pt')
        x = self.up1(x, x3)
        torch.save(x, res_path)

        res_path = join(save_path, 'up2.pt')
        x = self.up2(x, x2)
        torch.save(x, res_path)

        res_path = join(save_path, 'up3.pt')
        x = self.up3(x, x1)
        torch.save(x, res_path)

        res_path = join(save_path, 'up4.pt')
        x = self.up4(x, x0)
        torch.save(x, res_path)

        res_path = join(save_path, 'out.pt')
        out = self.out(x)
        torch.save(out, res_path)

        return out 


def plot(save_path: str) -> None: 
    lst = ['ip.pt', 'ch.pt', 'down1.pt', 'down2.pt', 'down3.pt', 'down4.pt',\
           'up1.pt', 'up2.pt', 'up3.pt', 'up4.pt', 'out.pt']
    for feat in lst: 
        unet_feat = join(save_path, 'unet', feat)
        q_unet_feat = join(save_path, 'q_unet', feat)
        unet_feat = torch.load(unet_feat)
        q_unet_feat= torch.load(q_unet_feat)
        unet_feat= rearrange(unet_feat, '1 c h w -> h w c')
        q_unet_feat= rearrange(q_unet_feat, '1 c h w -> h w c')
        kl_unet = torch.softmax(unet_feat, dim=-1) 
        kl_q_unet = torch.softmax(q_unet_feat, dim=-1)
        unet_feat = unet_feat/torch.linalg.norm(unet_feat, dim=-1).unsqueeze(-1) 
        q_unet_feat = q_unet_feat/torch.linalg.norm(q_unet_feat, dim=-1).unsqueeze(-1) 
        cos_sim = unet_feat * q_unet_feat
        cos_sim = torch.sum(cos_sim, dim=-1)
        kl_unet, kl_q_unet = kl_unet * torch.log(kl_unet/kl_q_unet), \
        kl_q_unet * torch.log(kl_q_unet/kl_unet)
        kl_unet = torch.sum(kl_unet, dim=-1)
        kl_q_unet = torch.sum(kl_q_unet, dim=-1)

        # ToPILImage()(cos_sim).save(join(save_path, str(ch) + '_' + feat[:-1] + 'ng'))
        ToPILImage()(cos_sim).save(join(save_path, 'cos_sim_' + feat[:-1] + 'ng'))
        ToPILImage()(kl_unet).save(join(save_path, 'kl_unet_'+feat[:-1] +'ng'))
        ToPILImage()(kl_q_unet).save(join(save_path, 'kl_q_unet_'+feat[:-1] +'ng'))

    os.system(f'rm -r {save_path}/unet/')
    os.system(f'rm -r {save_path}/q_unet/')
        
def concatenate(dir_paths: List[str]) -> None: 
    cos_sims = os.listdir(dir_paths[0])
    cos_sims = [cos_sim for cos_sim in cos_sims if cos_sim.startswith('cos')]
    save_path = dir_paths[0].split('/')[:-1]
    save_path = '/'.join(save_path)
    dir_paths = sorted(dir_paths, key=lambda dir_path: int(dir_path.split('/')[-1]))
    for cos_sim in cos_sims: 
        feat_lst = list()
        img_lst = list()
        mask_lst = list()
        kl_unet = list() 
        kl_q_unet = list() 
        for dir in dir_paths: 
            path = join(dir, cos_sim)
            img = Image.open(path) 
            img = ToTensor()(img) 
            _, h, w = img.shape
            feat_lst.append(img) 

            path = join(dir, 'img.png')
            img = Image.open(path) 
            img = ToTensor()(img) 
            img = Resize((h, w))(img)
            img_lst.append(img) 

            path = join(dir, 'mask.png')
            img = Image.open(path) 
            img = ToTensor()(img) 
            img = Resize((h, w))(img)
            mask_lst.append(img) 

            path = join(dir, 'kl_unet' + cos_sim[7:]) 
            img = Image.open(path) 
            img = ToTensor()(img) 
            img = Resize((h, w))(img)
            kl_unet.append(img) 

            path = join(dir, 'kl_q_unet' + cos_sim[7:]) 
            img = Image.open(path) 
            img = ToTensor()(img) 
            img = Resize((h, w))(img)
            kl_q_unet.append(img) 

        whole_lst = img_lst + kl_unet + kl_q_unet + feat_lst + mask_lst 
        res = torch.stack(whole_lst)
        feat_lst = make_grid(res, nrow=len(dir_paths))
        ToPILImage()(feat_lst).save(join(save_path, cos_sim[8:]))
    rm_dirs = os.listdir(save_path)
    for dir in rm_dirs: 
        dir = join(save_path, dir)
        if os.path.isdir(dir): 
            os.system(f'rm -r {dir}')







def main(args): 
    from PIL import Image 
    from torchvision.transforms import ToTensor 
    from tqdm import tqdm 

    unet_model = UNET().cuda() 
    ckpt = torch.load(args.unet_ckpt_path) 
    unet_model.load_state_dict(ckpt['model_state'])
    q_unet_model = Q_UNET().cuda() 
    ckpt = torch.load(args.q_unet_ckpt_path) 
    q_unet_model.load_state_dict(ckpt['model_state'])

    imgs = sorted(os.listdir('../res/images/'))
    masks = sorted(os.listdir('../res/masks/'))
    save_paths = list()
    for img_path, mask_path in tqdm(zip(imgs, masks)):
        idx = img_path[:-4]
        save_paths.append(join(args.save_path, idx))
        img = Image.open(join('../res/images/', img_path)).convert('L') 
        mask = Image.open(join('../res/masks/', mask_path))

        img = ToTensor()(img).unsqueeze(0).cuda()
        img += torch.randn_like(img) * float(args.save_path.split('/')[-1])
        mask = ToTensor()(mask) 
        save_path = join(args.save_path, idx)
        logits = unet_model(img, join(save_path, 'unet'))
        ToPILImage()(logits[0]).save(join(save_path, 'unet_pred.png')) 
        logits = q_unet_model(img, join(save_path, 'q_unet'))
        ToPILImage()(logits[0]).save(join(save_path, 'q_unet_pred.png')) 
        img = ToPILImage()(img[0]).save(join(save_path, 'img.png'))
        mask = ToPILImage()(mask).save(join(save_path, 'mask.png'))
        save_path = join(args.save_path, idx)
        plot(save_path)
    concatenate(save_paths)

        

if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--unet_ckpt_path', type=str)
    parser.add_argument('--q_unet_ckpt_path', type=str)
    parser.add_argument('--save_path',type=str)
    args = parser.parse_args() 
    print(f'{args = }')
    main(args)
