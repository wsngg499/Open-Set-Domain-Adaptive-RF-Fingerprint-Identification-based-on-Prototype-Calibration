import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.size(0),-1)

class resblocknet(nn.Module):
    def __init__(self, in_channels):
        super(resblocknet, self).__init__()

        F1= F2=(in_channels//4)
        self.conv2d_1 = nn.Conv1d(in_channels, F1, 1,bias=False)
        self.bn_1 = nn.BatchNorm1d(F1)
        self.relu_1=nn.LeakyReLU(0.2, inplace=True)

        self.conv2d_2 = nn.Conv1d(F1, F2,3, padding=1,bias=False)
        self.bn_2 = nn.BatchNorm1d(F2)
        self.relu_2=nn.LeakyReLU(0.2, inplace=True)

        self.conv2d_3 = nn.Conv1d(F2, in_channels,1,bias=False)
        self.bn_3 = nn.BatchNorm1d(in_channels)
        self.relu_3=nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x_shortcut = x
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.conv2d_3(x)
        x = self.bn_3(x)
        x += x_shortcut
        x = self.relu_3(x)
        return x


class mmblocknet(nn.Module):
    def __init__(self, in_channels):
        super(mmblocknet, self).__init__()

        F1= F2=(in_channels//4)

        self.gnet = nn.Sequential(
            nn.Conv1d(in_channels, F1, 1,bias=False),
            nn.BatchNorm1d(F1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(F1, F2,3, padding=1,bias=False),
            nn.BatchNorm1d(F2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(F2, in_channels,1,bias=False),
            nn.BatchNorm1d(in_channels),
        )
        self.relu_mm=nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x_shortcut = x
        x = self.gnet(x)
        x = torch.mul(x,x_shortcut)
        x = self.relu_mm(x)
        return x



class fouriernet(nn.Module):
    def __init__(self,inch,outch):
        super().__init__()
        self.conv1=nn.Conv1d(inch, outch, 3, 4, 1,bias=False)
        self.conv2=nn.Conv1d(inch, outch, 5, 4, 2,bias=False)
        self.conv3=nn.Conv1d(inch, outch, 7, 4, 3, bias=False)
        self.conv4 = nn.Conv1d(inch, outch, 9, 4, 4, bias=False)
        # self.w = Parameter((torch.rand([1, outch * 4, 1])), requires_grad=True)
    def forward(self,x):
        x0=self.conv1(x)
        x1=self.conv2(x)
        x2=self.conv3(x)
        x3=self.conv4(x)
        x=torch.cat((x0,x1,x2,x3),dim=-2)
        # x=x*self.w.expand(*x.size())
        return x

class activatenet(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.p0=Parameter(torch.rand([*size]),requires_grad=True)
        self.p1 =Parameter(torch.rand([*size]),requires_grad=True)

    def forward(self,x):
        p0=self.p0.expand(*x.size())
        p1 = self.p1.expand(*x.size())
        return (x*p0-p1)



class normnet(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.p0=Parameter(torch.rand([*size]),requires_grad=True)
        # self.p1 =Parameter(torch.rand([*size]),requires_grad=True)

    def forward(self,x):
        p0=self.p0.expand(*x.size())
        return (x-p0)
#
#
#
# class encodernet(nn.Module):
#     def __init__(self):
#         super(encodernet, self).__init__()
#         nc,ndf=2,24
#         if option.USE_MYACTIVATION:
#             self.main = nn.Sequential(
#                 activatenet((1, 1, 3040)),
#
#                 # nn.Conv1d(nc, ndf * 2, 3, 4, 1,bias=False),
#                 fouriernet(nc, ndf // 2),
#                 nn.BatchNorm1d(ndf * 2),
#                 activatenet((1, 48, 760)),
#                 # nn.ReLU(),
#                 # nn.LeakyReLU(0.2, inplace=True),
#                 mmblocknet(ndf * 2),
#
#                 # resblocknet(ndf * 2),
#
#                 nn.Conv1d(ndf * 2, ndf * 4, 9, 4,4,bias=False),
#                 # fouriernet(ndf * 2, ndf),
#                 nn.BatchNorm1d(ndf * 4),
#                 activatenet((1, 96, 190)),
#                 # nn.ReLU(),
#                 # nn.LeakyReLU(0.2, inplace=True),
#                 resblocknet(ndf * 4),
#
#                 nn.Conv1d(ndf * 4, ndf * 8, 9, 4, 4, bias=False),
#                 # fouriernet(ndf * 4, ndf * 2),
#                 nn.BatchNorm1d(ndf * 8),
#                 # directnet((1, 192, 48)),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 resblocknet(ndf * 8),
#
#                 nn.Conv1d(ndf * 8, ndf * 16, 9, 4, 4, bias=False),
#                 # fouriernet(ndf * 8, ndf * 4),
#                 nn.BatchNorm1d(ndf * 16),
#                 # directnet((1, 384, 12)),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # resblocknet(ndf * 16),
#
#                 Flatten(),
#             )
#         else:
#             self.main = nn.Sequential(
#                 # activatenet((1, 1, 3040)),
#
#                 nn.Conv1d(nc, ndf * 2, 7, 3, 2,bias=False),
#                 # fouriernet(nc, ndf // 2),
#                 nn.BatchNorm1d(ndf * 2),
#                 # activatenet((1, 48, 760)),
#                 # nn.ReLU(),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 resblocknet(ndf * 2),
#
#                 # resblocknet(ndf * 2),
#
#                 nn.Conv1d(ndf * 2, ndf * 4, 7, 3,2,bias=False),
#                 # fouriernet(ndf * 2, ndf),
#                 nn.BatchNorm1d(ndf * 4),
#                 # activatenet((1, 96, 190)),
#                 # nn.ReLU(),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 resblocknet(ndf * 4),
#
#                 nn.Conv1d(ndf * 4, ndf * 8, 7, 3 , 2, bias=False),
#                 # fouriernet(ndf * 4, ndf * 2),
#                 nn.BatchNorm1d(ndf * 8),
#                 # activatenet((1, 192, 48)),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 resblocknet(ndf * 8),
#
#                 nn.Conv1d(ndf * 8, ndf * 16,7, 3, 2, bias=False),
#                 # fouriernet(ndf * 8, ndf * 4),
#                 nn.BatchNorm1d(ndf * 16),
#                 # activatenet((1, 384, 12)),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # resblocknet(ndf * 16),
#
#                 Flatten(),
#             )
#
#     def forward(self, input):
#         output=self.main(input)
#         return output


class encodernet(nn.Module):
    def __init__(self):
        super(encodernet, self).__init__()
        nc,ndf=6,64

        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32

            nn.Conv1d(nc, ndf * 2, 4, 2, 2, bias=False),
            # fouriernet(nc,ndf//2) S,
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            resblocknet(ndf * 2),


            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            resblocknet(ndf * 4),
            # state size. (ndf*4) x 8 x 8
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 2, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            resblocknet(ndf * 8),
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(ndf *8,ndf, 4, 1, 2, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            resblocknet(ndf),
            nn.AdaptiveAvgPool1d(1),

            Flatten(),
        )

    def forward(self, input):
        output = self.main(input)
        return output

encoder=encodernet()
a=torch.ones([1,2,512])
#print(encoder(a).size())





class transnet(nn.Module):
    def __init__(self):
        super().__init__()
        numc=64
        self.gnet = nn.Sequential(
            nn.Linear(64,64,bias=True),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, numc, bias=True),
            #nn.BatchNorm1d(numc),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Softmax(-1),
        )

    def forward(self,x):
        x = self.gnet(x)
        return x

class classifynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor=encodernet()
        self.transfun=transnet()

    def forward(self,x):
        x = self.extractor(x)
        x=self.transfun(x)
        return x,x,x

class SimpleCNN1D(nn.Module):
    def __init__(self, num_features=64):
        super(SimpleCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_features)  # 输出64维特征

        # 兼容DANN：添加简单的domain_classifier（类似ResNet中的）
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)  # 二分类：源/目标域
        )

    def forward(self, x, alpha=0):
        # 特征提取
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feature = self.fc(out)

        # 兼容DANN：反转梯度 + 域分类（参考resnet_.py中的ReverseLayerF）
        from resnet_ import ReverseLayerF  # 假设resnet_.py中有这个
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return feature, domain_output, feature  # 兼容原有接口


class LSTM1D(nn.Module):
    def __init__(self, num_features=64, hidden_size=128, num_layers=5):  # 增加到5层
        super(LSTM1D, self).__init__()
        # 前置Conv1d提取局部特征，增加复杂度（类似ResNet初始层）
        self.initial_conv = nn.Conv1d(2, 64, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm1d(64)
        self.initial_relu = nn.LeakyReLU(0.2, inplace=True)

        # 5层双向LSTM，添加dropout和LayerNorm
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=0.2)

        # 后处理：GlobalAvgPool1d + 2层FC with 残差和LayerNorm
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size * 2, 256)  # 双向 *2
        self.fc2 = nn.Linear(256, num_features)
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(num_features)
        self.dropout = nn.Dropout(0.2)

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )

    def forward(self, x, alpha=0):
        # 前置Conv1d: (batch, 2, 512) -> (batch, 64, 512)
        out = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        # 转置为LSTM输入: (batch, 512, 64)
        out = out.permute(0, 2, 1)
        # LSTM: 5层双向
        out, (h_n, c_n) = self.lstm(out)
        # 拼接正向和反向最后隐藏状态: (batch, 256)
        feature = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        # 后处理: 2层FC with 残差
        residual = feature
        feature = F.relu(self.fc1(feature))
        feature = self.layer_norm1(feature)
        feature = self.dropout(feature)
        feature = self.fc2(feature + residual)  # 残差连接
        feature = self.layer_norm2(feature)

        from resnet_ import ReverseLayerF
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        return feature, domain_output, feature


class ViT1D(nn.Module):
    def __init__(self, patch_size=4, embed_dim=128, num_heads=8, num_layers=4, dim_feedforward=512, num_features=64,
                 dropout=0.2):
        super(ViT1D, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 时频特征提取
        self.fft_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, embed_dim, kernel_size=1)
        )
        self.time_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, embed_dim, kernel_size=1)
        )

        # Patch embedding
        self.patch_embed = nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=1000)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头：全局平均池化
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_features),
            nn.LayerNorm(num_features)
        )

        # DANN 域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2)
        )

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, alpha=0):
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        num_patches = seq_len // self.patch_size

        # 调试：打印形状
        # print(f"ViT1D input shape: {x.shape}, num_patches: {num_patches}")

        # 时频特征
        x_raw = x[:, 0, :] + 1j * x[:, 1, :]
        x_fft = torch.fft.fft(x_raw, dim=-1)
        fft_input = torch.stack([torch.real(x_fft), torch.imag(x_fft)], dim=1)
        fft_features = self.fft_conv(fft_input)  # (batch, embed_dim, seq_len)
        time_features = self.time_conv(x)  # (batch, embed_dim, seq_len)
        x = torch.cat([fft_features, time_features], dim=1)  # (batch, embed_dim*2, seq_len)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, num_patches)
        x = x.permute(0, 2, 1)  # (batch, num_patches, embed_dim)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer(x)  # (batch, num_patches, embed_dim)
        x = self.norm(x)

        # 全局平均池化
        feature = x.mean(dim=1)  # (batch, embed_dim)
        feature = self.fc(feature)  # (batch, num_features=64)

        # DANN
        from resnet_ import ReverseLayerF
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        return feature, domain_output, feature


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)