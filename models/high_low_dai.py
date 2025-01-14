import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取输入图像
image = cv2.imread(r'G:\Code\workspace\paper2\1715156515404.jpg', 0)  # 读取为灰度图像
# 将图像转换为PyTorch张量
image_tensor = torch.from_numpy(image).float()
image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度


class LearnableFilters(nn.Module):
    def __init__(self, dim, h, w):
        super(LearnableFilters, self).__init__()

        self.dim = dim
        self.h = h
        self.w = w

        # 初始化可学习的复权重
        self.complex_weight_low = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_band = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        # 对图像进行傅里叶变换
        fft_tensor = torch.fft.fft2(x)
        # 中心化处理
        fft_shift_tensor = torch.fft.fftshift(fft_tensor)

        # 将复权重转换为复数形式
        weight_low = torch.view_as_complex(self.complex_weight_low)
        weight_high = torch.view_as_complex(self.complex_weight_high)
        weight_band = torch.view_as_complex(self.complex_weight_band)

        # LJW
        _, _, h, w = fft_shift_tensor.shape
        ch, cw = h // 2, w // 2

        # 创建低通滤波器掩膜
        mask_low = torch.zeros_like(fft_shift_tensor)
        mask_low[:, :, ch - 30:ch + 31, cw - 30:cw + 31] = 1

        # 创建高通滤波器掩膜
        mask_high = torch.ones_like(fft_shift_tensor)
        mask_high[:, :, ch - 30:ch + 31, cw - 30:cw + 31] = 0

        # 创建带通滤波器掩膜
        mask_band = torch.zeros_like(fft_shift_tensor)
        mask_band[:, :, ch - 50:ch - 10, cw - 50:cw - 10] = 1
        mask_band[:, :, ch + 10:ch + 50, cw + 10:cw + 50] = 1

        # 应用低通滤波器
        fft_shift_low = fft_shift_tensor * mask_low
        fft_low = torch.fft.ifftshift(fft_shift_low)
        image_low = torch.fft.ifft2(fft_low)
        image_low = torch.abs(image_low).squeeze()

        # 应用高通滤波器
        fft_shift_high = fft_shift_tensor * mask_high
        fft_high = torch.fft.ifftshift(fft_shift_high)
        image_high = torch.fft.ifft2(fft_high)
        image_high = torch.abs(image_high).squeeze()

        # 应用带通滤波器
        fft_shift_band = fft_shift_tensor * mask_band
        fft_band = torch.fft.ifftshift(fft_shift_band)
        image_band = torch.fft.ifft2(fft_band)
        image_band = torch.abs(image_band).squeeze()

        # 对频域图像应用不同的滤波器
        fft_shift_low = image_low * weight_low
        fft_shift_high = image_high * weight_high
        fft_shift_band = image_band * weight_band

        # 逆傅里叶变换并获取幅值
        fft_low = torch.fft.ifftshift(fft_shift_low)
        image_low = torch.fft.ifft2(fft_low)
        image_low = torch.abs(image_low).squeeze()

        fft_high = torch.fft.ifftshift(fft_shift_high)
        image_high = torch.fft.ifft2(fft_high)
        image_high = torch.abs(image_high).squeeze()

        fft_band = torch.fft.ifftshift(fft_shift_band)
        image_band = torch.fft.ifft2(fft_band)
        image_band = torch.abs(image_band).squeeze()

        fft_shift_tensor = torch.fft.ifftshift(fft_shift_tensor)
        fft_shift_tensor = torch.fft.ifft2(fft_shift_tensor)
        fft_shift_tensor = torch.abs(fft_shift_tensor).squeeze()

        return fft_shift_tensor


# # 获取图像尺寸
# _, _, h, w = image_tensor.shape
#
# # 创建模型
# model = LearnableFilters(64, 256,256)
# x = torch.randn(8,64,256,256)
# # 将输入图像应用到模型中
# fft_shift_tensor, image_low, image_high, image_band = model(x)
# print(fft_shift_tensor.shape, image_low.shape, image_high.shape, image_band.shape)
# # 分离张量并转换为numpy数组
# image_low_np = image_low.detach().numpy()
# image_high_np = image_high.detach().numpy()
# image_band_np = image_band.detach().numpy()

# # 显示结果
# plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Input Image')
# plt.subplot(2, 3, 2), plt.imshow(20 * np.log(np.abs(fft_shift_tensor.squeeze().detach().numpy())), cmap='gray'), plt.title('FFT Image (Centered)')
# plt.subplot(2, 3, 4), plt.imshow(image_low_np, cmap='gray'), plt.title('Low-pass Filtered')
# plt.subplot(2, 3, 5), plt.imshow(image_high_np, cmap='gray'), plt.title('High-pass Filtered')
# plt.subplot(2, 3, 6), plt.imshow(image_band_np, cmap='gray'), plt.title('Band-pass Filtered')
# plt.tight_layout()
# plt.show()