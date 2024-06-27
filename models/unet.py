import torch 
from torchvision.transforms.functional import crop
import os 
import yaml
from einops import rearrange
from torch.nn import functional as F
from torch import bilinear, nn, threshold 
import pennylane as qml 
from faster_rcnn.model.faster_rcnn import FasterRCNN


class Quanv(nn.Module):
    def __init__(self, n_qubits, out):
        
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def q_all_kern(inputs, weights_0, weights_1, weights_2, weight_3, weight_4, weights_5):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights_0, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights_1, wires=range(n_qubits))
            qml.Rot(*weights_2, wires=0)
            qml.RY(weight_3, wires=1)
            qml.RZ(weight_4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Rot(*weights_5, wires=0)
            return [qml.expval(qml.Z(i)) for i in range(n_qubits)[-out:]]

        weight_shapes = {
            "weights_0": (3, n_qubits, 3),
            "weights_1": (3, n_qubits),
            "weights_2": 3,
            "weight_3": 1,
            "weight_4": (1,),
            "weights_5": 3,
        }

        init_method = {
            "weights_0": torch.nn.init.normal_,
            "weights_1": torch.nn.init.uniform_,
            "weights_2": torch.tensor([1., 2., 3.]),
            "weight_3": torch.tensor(1.),  # scalar when shape is not an iterable and is <= 1
            "weight_4": torch.tensor([1.]),
            "weights_5": torch.tensor([1., 2., 3.]),
        }
        super().__init__()
        self.fc1 = qml.qnn.TorchLayer(q_all_kern, weight_shapes=weight_shapes, init_method=init_method)
        self.fc3 = nn.Softmax(dim=-1)

        # fig, ax = qml.draw_mpl(q_all_kern ,expansion_strategy='device')(torch.randn(n_qubits),
        #                     torch.randn(3, n_qubits, 3), torch.randn(3, n_qubits), torch.randn(3,),
        #                     torch.randn(1,), torch.randn(1, ), torch.randn(3,))
        # fig.savefig('./model.png')

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__() 
        mid_ch = out_ch if not mid_ch else mid_ch
        self.net = nn.Sequential(nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(mid_ch),
                                 nn.ReLU(inplace=True),

                                 nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))

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

    def forward(self, x):
        x0 = self.ch(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.down4(x3)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        return self.out(x)

class Q_UNET(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, n_qubits=16, bilinear=True):
        super().__init__() 
        self.n_qubits = n_qubits
        self.ch = DoubleConv(in_ch, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.qml_encoder = nn.Conv2d(512, 1, 1, 1)
        self.qml_lay = Quanv(n_qubits, n_qubits)
        self.qml_decoder = nn.Conv2d(1, 512, 1, 1)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024//factor)

        self.up1 = UpBlock(1024, 512//factor, bilinear)
        self.up2 = UpBlock(512, 256//factor, bilinear)
        self.up3 = UpBlock(256, 128//factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.out = nn.Sequential(nn.Conv2d(64, out_ch, kernel_size=1, stride=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x0 = self.ch(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.down4(x3)
        x = self.qml_encoder(x)
        bs, c, h, w = x.shape
        x = x.reshape(bs, -1, self.n_qubits)
        x = x.reshape(bs, c, h, w)
        x = self.qml_decoder(x)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        return self.out(x)

class Small_UNET(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, bilinear=True):
        super().__init__() 
        self.ch = DoubleConv(in_ch, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 128)
        factor = 2 if bilinear else 1

        self.up3 = UpBlock(256, 128//factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.out = nn.Sequential(nn.Conv2d(64, out_ch, kernel_size=1, stride=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x0 = self.ch(x)
        x1 = self.down1(x0)
        x = self.down2(x1)

        x = self.up3(x, x1)
        x = self.up4(x, x0)

        return self.out(x)

class RCNN_UNET(nn.Module): 
    def __init__(self, in_ch=1, out_ch=1, threshold = 0.7): 
        super().__init__()
        self.unet = Small_UNET(in_ch, out_ch) 
        self.missed = 0
        f = open('./faster_rcnn/config/voc.yaml', 'r')
        config = yaml.safe_load(f)
        model_config = config['model_params']
        train_config = config['train_params']
        f.close()
        faster_rcnn_model = FasterRCNN(model_config, num_classes=2)
        faster_rcnn_model.eval()
        faster_rcnn_model.load_state_dict(torch.load(os.path.join('faster_rcnn', 
                                                                train_config['task_name'],
                                                                train_config['ckpt_name'])))
    
        faster_rcnn_model.roi_head.low_score_threshold = threshold
        self.faster_rcnn = faster_rcnn_model
        for p in self.faster_rcnn.parameters():
            p.requires_grad = False

    def corn_to_centre(self, box): 
        x1, y1, x2, y2 = box
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        w = y2-y1
        h = x2-x1
        cx = x1 + w//2 
        cy = y1 + h//2 
        cx = max(cx, 50)
        cy = max(cy, 50) 
        cx = min(cx, 398) 
        cy = min(cy, 270)
        x1 = cx - 50 
        x2 = cx + 50 
        y1 = cy - 50 
        y2 = cy + 50 
        return x1, y1, x2, y2

    
    def forward(self, x): 
        with torch.no_grad():
            _, frcnn_output = self.faster_rcnn(x, None)
            boxes = frcnn_output['boxes'] 
        if boxes.shape != (1, 4): 
            self.missed += 1
            res = self.unet(x) 
            return res 
        res = torch.zeros_like(x)
        x1, y1, x2, y2 = self.corn_to_centre(boxes[0])
        x = crop(x, x1, y1, x2-x1, y2-y1).detach()
        res[:, :, x1:x2, y1:y2] = self.unet(x)
        return res
 




if __name__ == '__main__': 
    torch.manual_seed(0)
    img = torch.randn(1, 1, 448, 320).float().cuda()
    # img = torch.randn(1, 1, 100, 100).float().cuda()
    # model = UNET(in_ch=2, out_ch=1).cuda() 
    model = RCNN_UNET(in_ch=1, out_ch=1).cuda() 
    model.train()
    
    img = img.cuda()
    logits = model(img.cuda())

    print(logits.shape)








