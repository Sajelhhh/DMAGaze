import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision.models import resnet18, resnet34
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EyesEncoder(nn.Module):
    def __init__(self):
        super(EyesEncoder, self).__init__()
        self.left_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.right_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 9 * 15, 128),  # Assuming input size (batch, 3, 36, 60)
            nn.ReLU(inplace=True),
            nn.Linear(128, 32)
        )

    def forward(self, left_eye, right_eye):
        left_feat = self.left_conv(left_eye)  # [8, 64, 9, 15]
        right_feat = self.right_conv(right_eye)

        combined_feat = torch.cat((left_feat, right_feat), dim=1)
        combined_feat = combined_feat.view(combined_feat.size(0), -1)  # Flatten
        output = self.fc(combined_feat)  # [b, 32]
        return output


class Disentangler(nn.Module):
    def __init__(self):
        super(Disentangler, self).__init__()
        self.k = None
        self.conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))

        if self.k is None:
            self.k = nn.Parameter(torch.randint(0, 2, x.shape).float()).to(device)
            # self.k = nn.Parameter(torch.randint(0, 2, x.shape).float())

        assert self.k.shape == x.shape, "Shape of k must match the shape of x"

        x_k = self.k * x
        x_m_k = (1 - self.k) * x  
        return x_k, x_m_k


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Output of the linear layer must match the size needed for the first ConvTranspose2d layer
        self.fc = nn.Linear(128, 512 * 7 * 7)  # Adjusting for 512 channels with 7x7 spatial dimensions
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # Upsample to 14x14
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Upsample to 28x28
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 56x56
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample to 112x112
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # Final upsample to 224x224

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 512, 7, 7)  # Reshape to match the input of the first deconv layer
        x = F.relu(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))  # Final activation to output image in [0, 1] range
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, out):
        # Channel attention
        out = out * self.channel_attention(out)

        # Spatial attention
        out = out * self.spatial_attention(out)
        return out


class NonLocalGaussianAttention(nn.Module):
    def __init__(self, in_dim):
        super(NonLocalGaussianAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        self.sigma = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def gaussian_kernel(self, q, k):
        q = q.permute(0, 2, 1)  # [b, h*w, c]
        k = k.permute(0, 2, 1)

        diff = q.unsqueeze(2) - k.unsqueeze(1)  
        dist_sq = torch.sum(diff ** 2, dim=-1)  
        similarity = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        return similarity

    def forward(self, x):
        batch_size, C, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width) 
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)  

        similarity = self.gaussian_kernel(query, key) 

        # Softmax normalization
        attention = self.softmax(similarity)

        # Weighted sum of values using attention map
        attention_out = torch.matmul(attention, value.permute(0, 2, 1))  
        attention_out = attention_out.view(batch_size, C, height, width)

        # Add residual connection
        out = x + attention_out
        return out


class MultiScaleFeatureFusionWithAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiScaleFeatureFusionWithAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_dim, out_dim, kernel_size=5, padding=2)
        self.non_local_attention = NonLocalGaussianAttention(out_dim)
        self.cbam = CBAM(out_dim)
        self.out_dim = out_dim
        self.tail = nn.Conv2d(self.out_dim * 2, self.out_dim, kernel_size=1)

    def forward(self, x):
        # Multi-scale convolutions
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)
        out = out1 + out2 + out3
        # Apply CBAM and Non-Local Attention
        cbam_out = self.cbam(out)
        non_local_out = self.non_local_attention(out)
        # Concatenate features from different heads
        out = torch.cat([cbam_out, non_local_out], dim=1)
        # Reduce dimension back to original
        out = self.tail(out)
        return out


class MultiScaleDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiScaleDepthwiseSeparableConv, self).__init__()

        # Multi-scale depthwise convolutions
        self.dw_conv1x1 = nn.Conv2d(in_dim, in_dim, kernel_size=1, groups=in_dim, padding=0)
        self.dw_conv3x3 = nn.Conv2d(in_dim, in_dim, kernel_size=3, groups=in_dim, padding=1)
        self.dw_conv5x5 = nn.Conv2d(in_dim, in_dim, kernel_size=5, groups=in_dim, padding=2)

        # Single pointwise convolutions
        self.pw_conv = nn.Conv2d(in_dim * 3, out_dim, kernel_size=1)

        # Spatial attention
        # self.spatial_attention = SpatialAttention()

        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dw_out1 = self.dw_conv1x1(x)
        dw_out2 = self.dw_conv3x3(x)
        dw_out3 = self.dw_conv5x5(x)

        out = torch.cat([dw_out1, dw_out2, dw_out3], dim=1)
        out = self.pw_conv(out)

        out = self.bn(out)
        out = self.relu(out)
        return out


class MultiScaleConvCBAMNonLocalDecouple(nn.Module):
    def __init__(self, num_per_epoch, in_dim, n):
        super(MultiScaleConvCBAMNonLocalDecouple, self).__init__()
        self.num_per_epoch = num_per_epoch
        self.in_dim = in_dim
        self.n = n
        self.sub_conv = nn.Conv2d(self.in_dim // self.n, self.in_dim // self.n, kernel_size=3, stride=1, padding=1)

        self.cbam_0 = CBAM(self.in_dim // self.n)  # out_dim
        self.cbam_1 = CBAM((self.in_dim // self.n) * 2)
        self.cbam_2 = CBAM((self.in_dim // self.n) * 3)
        self.cbam_3 = CBAM(self.in_dim)

        self.non_local_0 = NonLocalGaussianAttention(self.in_dim // self.n)
        self.non_local_1 = NonLocalGaussianAttention((self.in_dim // self.n) * 2)
        self.non_local_2 = NonLocalGaussianAttention((self.in_dim // self.n) * 3)
        self.non_local_3 = NonLocalGaussianAttention(self.in_dim)

        self.tail_0 = nn.Conv2d((self.in_dim // self.n) * 2, self.in_dim // self.n, kernel_size=1)
        self.tail_1 = nn.Conv2d((self.in_dim // self.n) * 4, (self.in_dim // self.n) * 2, kernel_size=1)
        self.tail_2 = nn.Conv2d((self.in_dim // self.n) * 6, (self.in_dim // self.n) * 3, kernel_size=1)
        self.tail_3 = nn.Conv2d(self.in_dim * 2, self.in_dim, kernel_size=1)

        self.non_eyes_conv = MultiScaleDepthwiseSeparableConv((self.in_dim // self.n) * 14, self.in_dim)
        # self.non_eyes_conv = nn.Conv2d((in_dim//n)*14, in_dim, kernel_size=1, stride=1)

    def forward(self, sub_s):
        for j in range(self.num_per_epoch):
            _sub_s = []
            if j == 0:
                p = sub_s
            else:
                c = q.size(1)
                p = torch.split(q, c // self.n, dim=1)
            for i, sub in enumerate(p):
                if i == 0:
                    sub_conv = self.sub_conv(sub)  # [b, 64, 7, 7]
                    a = self.cbam_0(sub_conv)
                    b = self.non_local_0(sub_conv)
                    _sub_s.append(self.tail_0(torch.cat((a, b), dim=1)))
                elif i == 1:
                    sub_conv = self.sub_conv(sub)
                    a = self.cbam_1(torch.cat((_sub_s[i - 1], sub_conv), dim=1))
                    b = self.non_local_1(torch.cat((_sub_s[i - 1], sub_conv), dim=1))
                    _sub_s.append(self.tail_1(torch.cat((a, b), dim=1)))
                elif i == 2:
                    sub_conv = self.sub_conv(sub)
                    a = self.cbam_2(torch.cat((_sub_s[i - 1], sub_conv), dim=1))
                    b = self.non_local_2(torch.cat((_sub_s[i - 1], sub_conv), dim=1))
                    _sub_s.append(self.tail_2(torch.cat((a, b), dim=1)))
                else:
                    sub_conv = self.sub_conv(sub)
                    a = self.cbam_3(torch.cat((_sub_s[i - 1], sub_conv), dim=1))
                    b = self.non_local_3(torch.cat((_sub_s[i - 1], sub_conv), dim=1))
                    _sub_s.append(self.tail_3(torch.cat((a, b), dim=1)))
            q = torch.cat(_sub_s, dim=1)  
            q = torch.cat((torch.cat(p, dim=1), q), dim=1)  
            q = self.non_eyes_conv(q) 
        return q


class NonEyesEncoder(nn.Module):
    def __init__(self):
        super(NonEyesEncoder, self).__init__()
        resnet = resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        # for layer in self.features[-1][-1].children():
        #     if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
        #         layer.stride = (1, 1)
        y = self.features(x)  
        self.save_feat_face = y
        # output = self.face_conv(y)
        return y


class HeadPoseEstimator(nn.Module):
    def __init__(self):
        super(HeadPoseEstimator, self).__init__()
        self.conv1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        # self.attention = SelfAttention(32)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 32)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.attention(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.eyes_encoder = EyesEncoder()
        self.non_eyes_encoder = NonEyesEncoder()
        self.head_pose_estimator = HeadPoseEstimator()

        self.disentangler = Disentangler()

        self.n = 4
        self.multiscaledecouple = MultiScaleConvCBAMNonLocalDecouple(4, 256, self.n)  # num_per_epoch, _, n

        self.linear1 = nn.Linear(256 * 7 * 7, 256)
        self.conv1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.linear2 = nn.Linear(256 * 7 * 7, 256)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        # self.vector_quantizer = VectorQuantizer(512, 160)
        self.decoder = Decoder()

        self.yaw_mlp = nn.Sequential(nn.Linear(32 + 128 + 32, 32), nn.Linear(32, 1))
        self.pitch_mlp = nn.Sequential(nn.Linear(32 + 128 + 32, 32), nn.Linear(32, 1))

    def forward(self, left_eye, right_eye, face):
        eyes_features = self.eyes_encoder(left_eye, right_eye)  

        face_features = self.non_eyes_encoder(face)  
        head_pose = self.head_pose_estimator(face_features) 

        x_1, x_2 = self.disentangler(face_features)  

        c_1, c_2 = x_1.size(1), x_2.size(1)
        sub_1s = torch.split(x_1, c_1 // self.n, dim=1)
        sub_2s = torch.split(x_2, c_2 // self.n, dim=1)

        # Multi-scale disentangle
        p_1 = self.multiscaledecouple(sub_1s)  
        p_2 = self.multiscaledecouple(sub_2s)  

        flatten_p_1 = p_1.view(p_1.size(0), -1)
        p_1 = self.linear1(flatten_p_1)
        p_1 = p_1.view(p_1.size(0), p_1.size(1), 1, 1)
        p_1 = self.conv1(p_1)
        p_1 = p_1.view(p_1.size(0), -1)  # [b, 128]

        flatten_p_2 = p_2.view(p_2.size(0), -1)
        p_2 = self.linear2(flatten_p_2)
        p_2 = p_2.view(p_2.size(0), p_2.size(1), 1, 1)
        p_2 = self.conv2(p_2)
        p_2 = p_2.view(p_2.size(0), -1)  # [b, 128]

        # reconstruction
        x_recon_1 = self.decoder(p_1)  
        x_recon_2 = self.decoder(p_2)  

        # gaze estimation
        input = torch.cat((eyes_features, p_1, head_pose), dim=1)  
        yaw = self.yaw_mlp(input)
        pitch = self.pitch_mlp(input)

        return yaw, pitch, x_recon_1, x_recon_2
