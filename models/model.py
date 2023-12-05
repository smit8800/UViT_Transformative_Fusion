import torch 
from encoder.vit import *  
from encoder.unet import *  

#just playing around a bit...don't mind me
class UTNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1, bilinear=False):
        super(UTNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.act = nn.Sigmoid()

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up1 = Up(128, 64, bilinear)

        self.inc = DoubleConv(3, 64)
        self.con1 = DoubleConv(128, 128)
        self.con2 = DoubleConv(256, 256)
        self.con3 = DoubleConv(512, 512)
        self.con4 = DoubleConv(1792, 1024)
        self.con5 = DoubleConv(512, 512)
        self.con6 = DoubleConv(256, 256)
        self.con7 = DoubleConv(128, 128)
        self.con8 = DoubleConv(64, 64)
        self.outconv = OutConv(64, 1)
        self.filter = OutConv(1,1)
        self.rectify = DoubleConv(768, 1024)

        self.rfb_u4 = RFB_modified(512, 512)
        self.rfb_u3 = RFB_modified(256, 256)
        self.rfb_u2 = RFB_modified(128, 128)
        self.rfb_u1 = RFB_modified(64, 64)

        self.tru = SETR_Naive()
        self.sig = nn.Sigmoid()

    def forward(self, x):


        x1 = self.inc(x)
        x1_s = x1

        x2 = self.down1(x1)
        x2 = self.con1(x2)
        x2_s = x2

        x3 = self.down2(x2)
        x3 = self.con2(x3)
        x3_s = x3

        x4 = self.down3(x3)
        x4 = self.con3(x4)
        x4_s = x4


        xb = self.down4(x4)
        xt = self.tru(x)
        #xt = self.rectify(xt)
        xb = torch.cat((xb, xt),1)
        xb = self.con4(xb)
        xb_s = xb


        y4 = self.up4(xb_s, x4_s)
        y4 = self.rfb_u4(y4)
        y4 = self.con5(y4)
        y4_s = y4

        y3 = self.up3(y4_s, x3_s)
        y3 = self.rfb_u3(y3)
        y3 = self.con6(y3)
        y3_s = y3

        y2 = self.up2(y3_s, x2_s)
        y2 = self.rfb_u2(y2)
        y2 = self.con7(y2)
        y2_s = y2

        y1 = self.up1(y2_s, x1_s)
        y1 = self.rfb_u1(y1)
        y1 = self.con8(y1)

        out = self.outconv(y1)
        #out = self.filter(out)
        out = self.act(out)

        return out