import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class LiDAR_EMB(nn.Module):
    def __init__(self, lidar_channel):
        super(LiDAR_EMB, self).__init__()
        self.conv_a1 = nn.Conv2d(lidar_channel, 10, (1, 1), stride=(1, 1), padding=0)
        self.conv_a2 = nn.Conv2d(lidar_channel, 20, (1, 1), stride=(1, 1), padding=0)
        self.conv_a3 = nn.Conv2d(lidar_channel, 30, (1, 1), stride=(1, 1), padding=0)
        self.BN_a1 = nn.BatchNorm2d(10)
        self.BN_a2 = nn.BatchNorm2d(20)
        self.BN_a3 = nn.BatchNorm2d(30)

        self.conv_b1 = nn.Conv2d(lidar_channel, 10, (1, 1), stride=(1, 1), padding=0)
        self.conv_b2 = nn.Conv2d(lidar_channel, 20, (1, 1), stride=(1, 1), padding=0)
        self.conv_b3 = nn.Conv2d(lidar_channel, 30, (1, 1), stride=(1, 1), padding=0)
        self.BN_b1 = nn.BatchNorm2d(10)
        self.BN_b2 = nn.BatchNorm2d(20)
        self.BN_b3 = nn.BatchNorm2d(30)

        self.conv_a = nn.Conv2d(60, 50, (1, 1), stride=(1, 1), padding=0)
        self.conv_b = nn.Conv2d(60, 50, (1, 1), stride=(1, 1), padding=0)
        self.BN_a = nn.BatchNorm2d(50)
        self.BN_b = nn.BatchNorm2d(50)

    def forward(self, x_in):
        """
        x_in: [b,c,h,w]
        """
        x_a1 = F.relu(self.BN_a1(self.conv_a1(x_in)))
        x_a2 = F.relu(self.BN_a2(self.conv_a2(x_in)))
        x_a3 = F.relu(self.BN_a3(self.conv_a3(x_in)))
        x_a = torch.cat([x_a1, x_a2, x_a3], 1)
        x_a = F.relu(self.BN_a(self.conv_a(x_a)))

        x_b1 = F.relu(self.BN_b1(self.conv_b1(x_in)))
        x_b2 = F.relu(self.BN_b2(self.conv_b2(x_in)))
        x_b3 = F.relu(self.BN_b3(self.conv_b3(x_in)))
        x_b = torch.cat([x_b1, x_b2, x_b3], 1)
        x_b = F.relu(self.BN_b(self.conv_b(x_b)))

        x_L = torch.cat([x_a, x_b], 1)

        return x_a, x_b, x_L


class HSI_EMB(nn.Module):
    def __init__(self, hsi_channel):
        super(HSI_EMB, self).__init__()
        self.conv_1 = nn.Conv2d(hsi_channel, 256, (1, 1), stride=(1, 1), padding=0)
        self.conv_2 = nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=1)
        self.conv_3 = nn.Conv2d(128, 100, (3, 3), stride=(1, 1), padding=1)
        self.BN_1 = nn.BatchNorm2d(256)
        self.BN_2 = nn.BatchNorm2d(128)
        self.BN_3 = nn.BatchNorm2d(100)

    def forward(self, x_in):
        """
        x_in: [b,c,h,w]
        """
        x_hsi = F.relu(self.BN_1(self.conv_1(x_in)))
        x_H = F.relu(self.BN_2(self.conv_2(x_hsi)))
        x_H = F.relu(self.BN_3(self.conv_3(x_H)))

        return x_H


class Feature_Interaction(nn.Module):
    def __init__(self):
        super(Feature_Interaction, self).__init__()
        self.avg_poolH = nn.AdaptiveAvgPool2d(1)
        self.max_poolH = nn.AdaptiveMaxPool2d(1)
        self.SQ_H = nn.Conv2d(100, 64, (1, 1), stride=(1, 1), padding=0)
        self.EX_H = nn.Conv2d(64, 100, (1, 1), stride=(1, 1), padding=0)
        self.avg_poolL = nn.AdaptiveAvgPool2d(1)
        self.max_poolL = nn.AdaptiveMaxPool2d(1)
        self.SQ_L = nn.Conv2d(100, 64, (1, 1), stride=(1, 1), padding=0)
        self.EX_L = nn.Conv2d(64, 100, (1, 1), stride=(1, 1), padding=0)

        self.ATT_SPA = nn.Conv2d(4, 1, (1, 1), stride=(1, 1), padding=0)

    def forward(self, x_H, x_L):
        """
        x_H,x_L: [b,c,h,w]
        """
        # channel attention
        avg_chnH = self.EX_H(F.relu(self.SQ_H(self.avg_poolH(x_H))))
        max_chnH = self.EX_H(F.relu(self.SQ_H(self.max_poolH(x_H))))
        chn_attH = F.softmax((avg_chnH + max_chnH), dim=1)

        avg_chnL = self.EX_L(F.relu(self.SQ_L(self.avg_poolL(x_L))))
        max_chnL = self.EX_L(F.relu(self.SQ_L(self.max_poolL(x_L))))
        chn_attL = F.softmax((avg_chnL + max_chnL), dim=1)

        x_Hc = x_H * chn_attH
        x_Lc = x_L * chn_attL
        x_Hc = x_Hc * chn_attL
        x_Lc = x_Lc * chn_attH

        # spatial attention
        avg_spaH = torch.mean(x_H, dim=1, keepdim=True)
        max_spaH, _ = torch.max(x_H, dim=1, keepdim=True)
        avg_spaL = torch.mean(x_L, dim=1, keepdim=True)
        max_spaL, _ = torch.max(x_L, dim=1, keepdim=True)
        spa = torch.cat([avg_spaH, max_spaH, avg_spaL, max_spaL], dim=1)
        spa_att = self.ATT_SPA(spa)
        b, c, h, w = spa_att.shape
        spa_att = spa_att.reshape(b, c, h * w)
        spa_att = F.softmax(spa_att, dim=2)
        spa_att = spa_att.reshape(b, c, h, w)
        x_Hs = x_H * spa_att
        x_Ls = x_L * spa_att

        #output
        H_out = x_H + x_Lc + x_Ls
        L_out = x_L + x_Hc + x_Hs

        return H_out, L_out


class Fusion_Unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Fusion_Unit, self).__init__()
        self.conv_1 = nn.Conv2d(dim_in, dim_out, (3, 3), stride=(1, 1), padding=0)
        self.BN_1 = nn.BatchNorm2d(dim_out)
        self.deconv = nn.ConvTranspose2d(dim_out, dim_in, stride=1, kernel_size=3, padding=0, output_padding=0) # output = (input-1)*stride + output_padding - 2padding + kernel_size
        self.BN_2 = nn.BatchNorm2d(dim_in)
        self.conv_2 = nn.Conv2d(dim_in, dim_out, (3, 3), stride=(1, 1), padding=0)
        self.BN_3 = nn.BatchNorm2d(dim_out)

    def forward(self, F_M):
        """
        F_M: [b,c,h,w]
        """
        x_sq = F.gelu(self.BN_1(self.conv_1(F_M)))
        x_ex = F.gelu(self.BN_2(self.deconv(x_sq)))
        Residual = F_M - x_ex
        x_r = F.gelu(self.BN_3(self.conv_2(Residual)))
        x_out = x_sq + x_r

        return x_out


class Cross_Fusion(nn.Module):
    def __init__(self, dim_head, heads, cls):
        super(Cross_Fusion, self).__init__()
        self.convH = nn.Conv2d(100, 128, (3, 3), stride=(1, 1), padding=0)
        self.BN_H1 = nn.BatchNorm2d(128)
        self.convL = nn.Conv2d(100, 128, (3, 3), stride=(1, 1), padding=0)
        self.BN_L1 = nn.BatchNorm2d(128)

        self.num_heads = heads
        self.dim_head = dim_head
        self.Hto_q = nn.Linear(128, dim_head * heads, bias=False)
        self.Hto_k = nn.Linear(128, dim_head * heads, bias=False)
        self.Hto_v = nn.Linear(128, dim_head * heads, bias=False)
        self.Lto_q = nn.Linear(128, dim_head * heads, bias=False)
        self.Lto_k = nn.Linear(128, dim_head * heads, bias=False)
        self.Lto_v = nn.Linear(128, dim_head * heads, bias=False)
        self.rescaleH = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescaleL = nn.Parameter(torch.ones(heads, 1, 1))
        self.projH = nn.Linear(dim_head * heads, 128, bias=True)
        self.projL = nn.Linear(dim_head * heads, 128, bias=True)
        self.LN_H2 = nn.LayerNorm(128)
        self.LN_L2 = nn.LayerNorm(128)

        self.FU_1 = Fusion_Unit(256, 360)
        self.FU_2 = Fusion_Unit(360, 512)
        self.FU_3 = Fusion_Unit(512, 512)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.FL = nn.Flatten()
        self.fcr = nn.Linear(512, cls)

    def forward(self, F_H, F_L):
        """
        F_H,F_L: [b,c,h,w]
        """
        #feature embedding
        F_H = F.relu(self.BN_H1(self.convH(F_H)))
        F_L = F.relu(self.BN_L1(self.convH(F_L)))

        #stage 1 for feature cross
        b, c, h, w = F_H.shape
        F_H = F_H.permute(0, 2, 3, 1)
        F_L = F_L.permute(0, 2, 3, 1)
        F_H = F_H.reshape(b, h * w, c)
        F_L = F_L.reshape(b, h * w, c)

        Hq_inp = self.Hto_q(F_H)
        Hk_inp = self.Hto_k(F_H)
        Hv_inp = self.Hto_v(F_H)
        Hq, Hk, Hv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
                      (Hq_inp, Hk_inp, Hv_inp))  # 对qkv调整形状
        Lq_inp = self.Lto_q(F_L)
        Lk_inp = self.Lto_k(F_L)
        Lv_inp = self.Lto_v(F_L)
        Lq, Lk, Lv = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=self.num_heads),
                      (Lq_inp, Lk_inp, Lv_inp))

        Hq = F.normalize(Hq, dim=-2, p=2)
        Hk = F.normalize(Hk, dim=-2, p=2)
        Lq = F.normalize(Lq, dim=-2, p=2)
        Lk = F.normalize(Lk, dim=-2, p=2)

        attnH = (Hk.transpose(-2, -1) @ Lq)
        attnH = attnH * self.rescaleH
        attnH = attnH.softmax(dim=-1)
        attnL = (Lk.transpose(-2, -1) @ Hq)
        attnL = attnL * self.rescaleL
        attnL = attnL.softmax(dim=-1)

        x_H = Hv @ attnH  # x_H:b,heads,hw,d
        x_L = Lv @ attnL

        x_H = x_H.permute(0, 2, 1, 3)  # x_H:b,hw,heads,d
        x_H = x_H.reshape(b, h * w, self.num_heads * self.dim_head)
        out_H = self.projH(x_H)  # out_H:b,hw,c
        x_L = x_L.permute(0, 2, 1, 3)
        x_L = x_L.reshape(b, h * w, self.num_heads * self.dim_head)
        out_L = self.projL(x_L)

        F_H = F_H + out_H
        F_L = F_L + out_L

        F_H = F_H.reshape(b, h, w, c)
        F_H = self.LN_H2(F_H)
        F_H = F_H.permute(0, 3, 1, 2) # F_H:b,c,h,w
        F_L = F_L.reshape(b, h, w, c)
        F_L = self.LN_L2(F_L)
        F_L = F_L.permute(0, 3, 1, 2)

        F_M = torch.cat([F_H, F_L], axis=1)

        # stage 2 for feature cross
        F_M = self.FU_1(F_M)
        F_M = self.FU_2(F_M)
        F_M = self.FU_3(F_M)

        # classification
        F_M = self.GAP(F_M)
        x = self.FL(F_M)
        x_result = F.softmax((self.fcr(x)), dim=1)

        return x_result


class model(nn.Module):
    def __init__(self, hsi_channel, lidar_channel, cls):
        super(model, self).__init__()
        self.L_EMB = LiDAR_EMB(lidar_channel)
        self.H_EMB = HSI_EMB(hsi_channel)
        self.Feature_Interaction = Feature_Interaction()
        self.Cross_Fusion = Cross_Fusion(128, 3, cls)
    def forward(self, x_H, x_L):

        x_H = x_H.permute(0, 3, 1, 2)
        x_L = x_L.permute(0, 3, 1, 2)

        x_H= self.H_EMB(x_H)
        x_a, x_b, x_L = self.L_EMB(x_L)

        F_H, F_L = self.Feature_Interaction(x_H, x_L)

        result = self.Cross_Fusion(F_H, F_L)

        return x_H, x_a, x_b, x_L, result