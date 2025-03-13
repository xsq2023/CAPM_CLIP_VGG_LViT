import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
import clip  

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        print(f"ConvBatchNorm output shape: {out.shape}")
        return out


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pix_level = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        if skip_x is not None:
            x = torch.cat([skip_x, up], dim=1)
        else:
            x=up
        x = self.nConvs(x)
        x = self.pix_level(x) 
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model="ViT-B/32"):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model, device="cuda")
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.text_proj = nn.ModuleDict({
            'stage1': nn.Sequential(nn.Linear(512, 128)),  
            'stage2': nn.Sequential(nn.Linear(512, 256)), 
            'stage3': nn.Sequential(nn.Linear(512, 512)), 
            'stage4': nn.Sequential(nn.Linear(512, 1024)) 
        })
        self.compress = nn.ModuleDict({
            'stage1': nn.Linear(128, 64),
            'stage2': nn.Linear(256, 128),
            'stage3': nn.Linear(512, 256),
            'stage4': nn.Linear(1024, 512)
        })

    def forward(self, text, stage):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        expanded = self.text_proj[stage](text_features.float())  
        compressed = self.compress[stage](expanded)              
        print(f"TextEncoder output shape ({stage}): {compressed.shape}")
        return compressed  

class TextGuidedCrossAttention(nn.Module):
    def __init__(self, in_channels, hw):
        super().__init__()
        self.l = nn.Parameter(torch.randn(1, hw)) 

        self.G = nn.Sequential(
            nn.Linear(in_channels, 2*in_channels), 
            nn.LayerNorm(2*in_channels),
            nn.GELU(),
            nn.Linear(2*in_channels, in_channels)    
        )
        
        self.conv_q = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_k = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.d_k = in_channels // 8  

    def forward(self, img_feat, text_feat):
        B, C, H, W = img_feat.shape
        GvT = self.G(text_feat)  
        F_T = torch.matmul(GvT.unsqueeze(2), self.l)  
        F_T = F_T.view(B, C, H, W)  
        

        Q = self.conv_q(img_feat).view(B, -1, H*W)  
        K = self.conv_k(F_T).view(B, -1, H*W)       
        V = self.conv_v(F_T).view(B, -1, H*W)       
        

        energy = torch.bmm(Q.permute(0,2,1), K) 
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(V, attention.permute(0,2,1))
        out = out.view(B, C, H, W)
        
        print(f"TextGuidedCrossAttention output shape: {out.shape}")
        return img_feat + self.gamma * out


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16_bn(pretrained=True)

        self.stage1 = nn.Sequential(*vgg.features[:7])   
        self.stage2 = nn.Sequential(*vgg.features[7:14]) 
        self.stage3 = nn.Sequential(*vgg.features[14:24]) 
        self.stage4 = nn.Sequential(*vgg.features[24:34]) 
        for param in self.stage1.parameters():
            param.requires_grad = False
        for param in self.stage2.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x1 = self.stage1(x)  
        print(f"VGG stage1 output shape: {x1.shape}")
        x2 = self.stage2(x1) 
        print(f"VGG stage2 output shape: {x2.shape}")
        x3 = self.stage3(x2) 
        print(f"VGG stage3 output shape: {x3.shape}")
        x4 = self.stage4(x3) 
        print(f"VGG stage4 output shape: {x4.shape}")
        return [x1, x2, x3, x4]


class CAPM_LViT(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.text_encoder = CLIPTextEncoder()
        self.img_encoder = VGGEncoder()
      
        
        self.cross_att1 = TextGuidedCrossAttention(64, hw=112*112)   
        self.cross_att2 = TextGuidedCrossAttention(128, hw=56*56)  
        self.cross_att3 = TextGuidedCrossAttention(256, hw=28*28)   
        self.cross_att4 = TextGuidedCrossAttention(512, hw=14*14)  
      
        
        self.up4 = UpblockAttention(512 + 256, 256, nb_Conv=2)  
        self.up3 = UpblockAttention(256 + 128, 128, nb_Conv=2)   
        self.up2 = UpblockAttention(128 + 64, 64, nb_Conv=2)    
        self.up1 = UpblockAttention(64, 64, nb_Conv=2)         
        self.outc = nn.Conv2d(64, n_classes, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, text_tokens):
   
        feats = self.img_encoder(x)
        x1, x2, x3, x4 = feats
        
       
        text_feat1 = self.text_encoder(text_tokens, 'stage1')  
        text_feat2 = self.text_encoder(text_tokens, 'stage2')  
        text_feat3 = self.text_encoder(text_tokens, 'stage3')  
        text_feat4 = self.text_encoder(text_tokens, 'stage4')  
        
  
        x1 = self.cross_att1(x1, text_feat1)
        x2 = self.cross_att2(x2, text_feat2)
        x3 = self.cross_att3(x3, text_feat3)
        x4 = self.cross_att4(x4, text_feat4)
        
    
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, None)
        
        out = self.outc(x)
        return self.sigmoid(out)



if __name__ == '__main__':


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

   
    x = torch.randn(1, 3, 224, 224).to(device)

  
    text_tokens = clip.tokenize(["a photo of a cat"]).to(device)

    
    model = CAPM_LViT(n_channels=3, n_classes=1).to(device)
    model.eval()


    with torch.no_grad():
        output = model(x, text_tokens)

 
    print("Output shape:", output.shape)
