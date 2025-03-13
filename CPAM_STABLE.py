import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
import clip  # 需要安装openai-clip包

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
        # 添加PixLevelModule
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
        x = self.pix_level(x)  # 细化局部特征
        return x


class CLIPTextEncoder(nn.Module):
    """CLIP文本编码器（按论文调整投影层）"""
    def __init__(self, clip_model="ViT-B/32"):
        super().__init__()
        self.clip_model, _ = clip.load(clip_model, device="cuda")
        # 冻结CLIP底层参数（按论文冻结底层+微调顶层）
        for param in self.clip_model.parameters():
            param.requires_grad = False
        # 可微调投影层（调整为匹配各stage的2C→C）
        self.text_proj = nn.ModuleDict({
            'stage1': nn.Sequential(nn.Linear(512, 128)),  # 2C=128 for C=64
            'stage2': nn.Sequential(nn.Linear(512, 256)), # 2C=256 for C=128
            'stage3': nn.Sequential(nn.Linear(512, 512)), # 2C=512 for C=256
            'stage4': nn.Sequential(nn.Linear(512, 1024)) # 2C=1024 for C=512
        })
        # 共享的压缩层（2C → C）
        self.compress = nn.ModuleDict({
            'stage1': nn.Linear(128, 64),
            'stage2': nn.Linear(256, 128),
            'stage3': nn.Linear(512, 256),
            'stage4': nn.Linear(1024, 512)
        })

    def forward(self, text, stage):
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)
        # 动态选择投影层
        expanded = self.text_proj[stage](text_features.float())  # 2C
        compressed = self.compress[stage](expanded)              # C
        print(f"TextEncoder output shape ({stage}): {compressed.shape}")
        return compressed  # [B, C]

class TextGuidedCrossAttention(nn.Module):
    """修正后的CPAM-TG模块（按论文公式）"""
    def __init__(self, in_channels, hw):
        super().__init__()
        # 可学习参数l ∈ R^{1×HW}（根据各stage的HW动态初始化）
        self.l = nn.Parameter(torch.randn(1, hw))  # !!! 修正l的维度 !!!
        
        # 文本特征处理（动态适配各stage的C）
        self.G = nn.Sequential(
            nn.Linear(in_channels, 2*in_channels),  # 按论文先扩展到2C
            nn.LayerNorm(2*in_channels),
            nn.GELU(),
            nn.Linear(2*in_channels, in_channels)    # 再压缩到C
        )
        
        # PAM参数
        self.conv_q = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_k = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.d_k = in_channels // 8  # 用于缩放注意力

    def forward(self, img_feat, text_feat):
        B, C, H, W = img_feat.shape
        # 文本特征转换（严格按论文公式）
        GvT = self.G(text_feat)  # [B, C]
        F_T = torch.matmul(GvT.unsqueeze(2), self.l)  # [B, C, HW] (按论文的矩阵乘法)
        F_T = F_T.view(B, C, H, W)  # 重塑为特征图
        
        # 生成Q/K/V
        Q = self.conv_q(img_feat).view(B, -1, H*W)  # [B, C//8, HW]
        K = self.conv_k(F_T).view(B, -1, H*W)       # [B, C//8, HW]
        V = self.conv_v(F_T).view(B, -1, H*W)       # [B, C, HW]
        
        # 注意力计算（增加缩放因子）
        energy = torch.bmm(Q.permute(0,2,1), K) #/ (self.d_k ** 0.5)  # !!! 缩放 !!!
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(V, attention.permute(0,2,1))
        out = out.view(B, C, H, W)
        
        print(f"TextGuidedCrossAttention output shape: {out.shape}")
        return img_feat + self.gamma * out


class VGGEncoder(nn.Module):
    """ImageNet预训练的VGG16作为图像编码器"""
    def __init__(self):
        super().__init__()
        vgg = vgg16_bn(pretrained=True)
        # 提取特征阶段
        self.stage1 = nn.Sequential(*vgg.features[:7])   # 64通道
        self.stage2 = nn.Sequential(*vgg.features[7:14]) # 128
        self.stage3 = nn.Sequential(*vgg.features[14:24]) # 256
        self.stage4 = nn.Sequential(*vgg.features[24:34]) # 512
        # 冻结前3个stage
        for param in self.stage1.parameters():
            param.requires_grad = False
        for param in self.stage2.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x1 = self.stage1(x)  # [B,64,224,224]
        print(f"VGG stage1 output shape: {x1.shape}")
        x2 = self.stage2(x1) # [B,128,112,112]
        print(f"VGG stage2 output shape: {x2.shape}")
        x3 = self.stage3(x2) # [B,256,56,56]
        print(f"VGG stage3 output shape: {x3.shape}")
        x4 = self.stage4(x3) # [B,512,28,28]
        print(f"VGG stage4 output shape: {x4.shape}")
        return [x1, x2, x3, x4]


class CAPM_LViT(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.text_encoder = CLIPTextEncoder()
        self.img_encoder = VGGEncoder()
      
        # 修正各stage的CPAM-TG的hw参数（按实际特征图尺寸）
        self.cross_att1 = TextGuidedCrossAttention(64, hw=112*112)   # stage1实际输出112x112
        self.cross_att2 = TextGuidedCrossAttention(128, hw=56*56)    # stage2输出56x56
        self.cross_att3 = TextGuidedCrossAttention(256, hw=28*28)    # stage3输出28x28
        self.cross_att4 = TextGuidedCrossAttention(512, hw=14*14)    # stage4输出14x14
      
        # 解码器调整输入通道数
        self.up4 = UpblockAttention(512 + 256, 256, nb_Conv=2)  # x4 (512) + x3 (256) → 768
        self.up3 = UpblockAttention(256 + 128, 128, nb_Conv=2)   # up4输出256 + x2 (128) → 384
        self.up2 = UpblockAttention(128 + 64, 64, nb_Conv=2)     # up3输出128 + x1 (64) → 192
        self.up1 = UpblockAttention(64, 64, nb_Conv=2)           # 无跳跃连接
        self.outc = nn.Conv2d(64, n_classes, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, text_tokens):
        # 图像编码
        feats = self.img_encoder(x)
        x1, x2, x3, x4 = feats
        
        # 文本编码（分stage获取对应投影）
        text_feat1 = self.text_encoder(text_tokens, 'stage1')  # [B,64]
        text_feat2 = self.text_encoder(text_tokens, 'stage2')  # [B,128]
        text_feat3 = self.text_encoder(text_tokens, 'stage3')  # [B,256]
        text_feat4 = self.text_encoder(text_tokens, 'stage4')  # [B,512]
        
        # 跨模态注意力（各stage使用对应文本特征）
        x1 = self.cross_att1(x1, text_feat1)
        x2 = self.cross_att2(x2, text_feat2)
        x3 = self.cross_att3(x3, text_feat3)
        x4 = self.cross_att4(x4, text_feat4)
        
        # 解码过程保持不变
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, None)
        
        out = self.outc(x)
        return self.sigmoid(out)



if __name__ == '__main__':

    # 根据是否有 CUDA 可用选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 构造一个伪造的输入图像，尺寸为 [B, C, H, W] = [1, 3, 224, 224]
    x = torch.randn(1, 3, 224, 224).to(device)

    # 使用 clip.tokenize 对文本进行编码（返回的 tokens 张量形状通常为 [B, token_length]）
    text_tokens = clip.tokenize(["a photo of a cat"]).to(device)

    # 实例化模型，并将其移动到指定设备上
    model = CAPM_LViT(n_channels=3, n_classes=1).to(device)
    model.eval()  # 设为评估模式

    # 前向传播测试
    with torch.no_grad():
        output = model(x, text_tokens)

    # 输出模型预测结果的尺寸
    print("Output shape:", output.shape)
