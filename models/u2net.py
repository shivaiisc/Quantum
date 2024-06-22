from torch import nn 
import torch 
from torch.nn import functional as F 

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super().__init__() 

        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, 
                                           padding=1*dirate,dilation=1*dirate),
                                 nn.BatchNorm2d(out_ch),
                                 nn.Dropout2d(p=0.36),
                                 nn.ReLU(inplace=True))
    
    def forward(self, x):
        return self.net(x)

def up_sample(inp, f_like):
    return F.interpolate(inp, size=f_like.shape[2:], mode='bilinear') 

class ResBlock7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        
        self.input = Block(in_ch, out_ch)
        self.down0 = Block(out_ch, mid_ch)
        self.down1 = Block(mid_ch, mid_ch)
        self.down2 = Block(mid_ch, mid_ch)
        self.down3 = Block(mid_ch, mid_ch)
        self.down4 = Block(mid_ch, mid_ch)
        self.down5 = Block(mid_ch, mid_ch)
        self.down6 = Block(mid_ch, mid_ch, dirate=2)

        self.up5 = Block(2*mid_ch, mid_ch)
        self.up4 = Block(2*mid_ch, mid_ch)
        self.up3 = Block(2*mid_ch, mid_ch)
        self.up2 = Block(2*mid_ch, mid_ch)
        self.up1 = Block(2*mid_ch, mid_ch)
        self.up0 = Block(2*mid_ch, out_ch)

        self.pool =nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

    def forward(self, x):
        xinput = self.input(x) 

        down0 = self.down0(xinput)
        down1 = self.down1(self.pool(down0)) 
        down2 = self.down2(self.pool(down1)) 
        down3 = self.down3(self.pool(down2))
        down4 = self.down4(self.pool(down3))
        down5 = self.down5(self.pool(down4)) 
        out = self.down6(down5) 

        out = up_sample(self.up5(torch.cat([out, down5], dim=1)), down4) 
        out = up_sample(self.up4(torch.cat([out, down4], dim=1)), down3)
        out = up_sample(self.up3(torch.cat([out, down3], dim=1)), down2) 
        out = up_sample(self.up2(torch.cat([out, down2], dim=1)), down1) 
        out = up_sample(self.up1(torch.cat([out, down1], dim=1)), down0)
        out = self.up0(torch.cat([out, down0], dim=1))

        return out + xinput 

class ResBlock6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        
        self.input = Block(in_ch, out_ch)
        self.down0 = Block(out_ch, mid_ch)
        self.down1 = Block(mid_ch, mid_ch)
        self.down2 = Block(mid_ch, mid_ch)
        self.down3 = Block(mid_ch, mid_ch)
        self.down4 = Block(mid_ch, mid_ch)
        self.down5 = Block(mid_ch, mid_ch, dirate=2)

        self.up4 = Block(2*mid_ch, mid_ch)
        self.up3 = Block(2*mid_ch, mid_ch)
        self.up2 = Block(2*mid_ch, mid_ch)
        self.up1 = Block(2*mid_ch, mid_ch)
        self.up0 = Block(2*mid_ch, out_ch)

        self.pool =nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

    def forward(self, x):
        xinput = self.input(x) 

        down0 = self.down0(xinput)
        down1 = self.down1(self.pool(down0)) 
        down2 = self.down2(self.pool(down1)) 
        down3 = self.down3(self.pool(down2))
        down4 = self.down4(self.pool(down3))
        out = self.down5(down4) 

        out = up_sample(self.up4(torch.cat([out, down4], dim=1)), down3)
        out = up_sample(self.up3(torch.cat([out, down3], dim=1)), down2) 
        out = up_sample(self.up2(torch.cat([out, down2], dim=1)), down1) 
        out = up_sample(self.up1(torch.cat([out, down1], dim=1)), down0)
        out = self.up0(torch.cat([out, down0], dim=1))

        return out + xinput 


class ResBlock5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__() 
    
        self.input = Block(in_ch, out_ch)
        self.down0 = Block(out_ch, mid_ch)
        self.down1 = Block(mid_ch, mid_ch)
        self.down2 = Block(mid_ch, mid_ch)
        self.down3 = Block(mid_ch, mid_ch)
        self.down4 = Block(mid_ch, mid_ch, dirate=2)

        self.up3 = Block(2*mid_ch, mid_ch)
        self.up2 = Block(2*mid_ch, mid_ch)
        self.up1 = Block(2*mid_ch, mid_ch)
        self.up0 = Block(2*mid_ch, out_ch)

        self.pool =nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

    def forward(self, x):
        xinput = self.input(x) 

        down0 = self.down0(xinput)
        down1 = self.down1(self.pool(down0)) 
        down2 = self.down2(self.pool(down1)) 
        down3 = self.down3(self.pool(down2))
        out = self.down4(down3)

        out = up_sample(self.up3(torch.cat([out, down3], dim=1)), down2) 
        out = up_sample(self.up2(torch.cat([out, down2], dim=1)), down1) 
        out = up_sample(self.up1(torch.cat([out, down1], dim=1)), down0)
        out = self.up0(torch.cat([out, down0], dim=1))

        return out + xinput 

class ResBlock4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__() 
    
        self.input = Block(in_ch, out_ch)
        self.down0 = Block(out_ch, mid_ch)
        self.down1 = Block(mid_ch, mid_ch)
        self.down2 = Block(mid_ch, mid_ch)
        self.down3 = Block(mid_ch, mid_ch, dirate=2)

        self.up2 = Block(2*mid_ch, mid_ch)
        self.up1 = Block(2*mid_ch, mid_ch)
        self.up0 = Block(2*mid_ch, out_ch)

        self.pool =nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

    def forward(self, x):
        xinput = self.input(x) 

        down0 = self.down0(xinput)
        down1 = self.down1(self.pool(down0)) 
        down2 = self.down2(self.pool(down1)) 
        out = self.down3(down2)

        out = up_sample(self.up2(torch.cat([out, down2], dim=1)), down1) 
        out = up_sample(self.up1(torch.cat([out, down1], dim=1)), down0)
        out = self.up0(torch.cat([out, down0], dim=1))

        return out + xinput 



class ResBlock4Full(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__() 
    
        self.input = Block(in_ch, out_ch)
        self.down0 = Block(out_ch, mid_ch)
        self.down1 = Block(mid_ch, mid_ch, dirate=2)
        self.down2 = Block(mid_ch, mid_ch, dirate=4)
        self.down3 = Block(mid_ch, mid_ch, dirate=8)

        self.up2 = Block(2*mid_ch, mid_ch, dirate=4)
        self.up1 = Block(2*mid_ch, mid_ch, dirate=2)
        self.up0 = Block(2*mid_ch, out_ch)


    def forward(self, x):
        xinput = self.input(x) 

        down0 = self.down0(xinput)
        down1 = self.down1(down0) 
        down2 = self.down2(down1) 
        out = self.down3(down2)

        out = up_sample(self.up2(torch.cat([out, down2], dim=1)), down1) 
        out = up_sample(self.up1(torch.cat([out, down1], dim=1)), down0)
        out = self.up0(torch.cat([out, down0], dim=1))

        return out + xinput 


class U2NET(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__() 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.unet_enc0 = ResBlock7(in_ch, 32, 64)
        self.unet_enc1 = ResBlock6(64, 32, 128)
        self.unet_enc2 = ResBlock5(128, 64, 256) 
        self.unet_enc3 = ResBlock4(256, 128, 512)
        self.unet_enc4 = ResBlock4Full(512, 256, 512)
        self.unet5 = ResBlock4Full(512, 256, 512)

        self.unet_dec4 = ResBlock4Full(1024, 256, 512)
        self.unet_dec3 = ResBlock4(1024, 128, 256)
        self.unet_dec2 = ResBlock5(512, 64, 128)
        self.unet_dec1 = ResBlock6(256, 32, 64)
        self.unet_dec0 = ResBlock7(128, 16, 64)

        self.out0 = nn.Conv2d(64,out_ch, kernel_size=3, padding=1)
        self.out1 = nn.Conv2d(64,out_ch, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(128,out_ch, kernel_size=3, padding=1)
        self.out3 = nn.Conv2d(256,out_ch, kernel_size=3, padding=1)
        self.out4 = nn.Conv2d(512,out_ch, kernel_size=3, padding=1)
        self.out5 = nn.Conv2d(512,out_ch, kernel_size=3, padding=1)

        self.res = nn.Sequential(nn.Conv2d(6*out_ch, out_ch, kernel_size=1),
                                 nn.Sigmoid())
        

    def forward(self, x):
        
        enc0 = self.unet_enc0(x)
        enc1 = self.unet_enc1(self.pool(enc0))
        enc2 = self.unet_enc2(self.pool(enc1))
        enc3 = self.unet_enc3(self.pool(enc2)) 
        enc4 = self.unet_enc4(self.pool(enc3))
        enc5 = self.unet5(self.pool(enc4)) 

        dec4 = self.unet_dec4(torch.cat([up_sample(enc5, enc4), enc4], dim=1))
        dec3 = self.unet_dec3(torch.cat([up_sample(dec4, enc3), enc3], dim=1))
        dec2 = self.unet_dec2(torch.cat([up_sample(dec3, enc2), enc2], dim=1))
        dec1 = self.unet_dec1(torch.cat([up_sample(dec2, enc1), enc1], dim=1))
        dec0 = self.unet_dec0(torch.cat([up_sample(dec1, enc0), enc0], dim=1))
        
        out0 = self.out0(dec0)
        out1 = up_sample(self.out1(dec1), out0)
        out2 = up_sample(self.out2(dec2), out0)
        out3 = up_sample(self.out3(dec3), out0)
        out4 = up_sample(self.out4(dec4), out0)
        out5 = up_sample(self.out5(enc5), out0) 

        res = self.res(torch.cat([out0, out1, out2, out3, out4, out5], dim=1)) 

        return res 










if __name__ == '__main__': 
    from torchsummary import summary
    from PIL import Image 
    from torchvision.transforms import ToTensor
    
    torch.manual_seed(0)
    res4 = U2NET(in_ch=1).cuda()

    # print(summary(rsu4, (1, 444, 342)))
    # print(summary(res4, (1, 444, 342)))
    img = Image.open('../data/0/img_111.jpg').convert('L')
    img = ToTensor()(img).unsqueeze(0).cuda()
    out1 = res4(img) 
    print(out1.shape)







