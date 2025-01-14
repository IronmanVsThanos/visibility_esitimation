# coding=utf-8
import cv2
import numpy as np
import math
import torch


# 图像基本特征计算函数
def cal_contrast(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m, n = img1.shape
    img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE) / 1.0
    rows_ext, cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)
    fco = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)
    fco = fco / (m * n)
    return round(fco, 6)


def cal_bright(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return v.mean() / 255.0


def Histogram_cal(img):
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    hist = cv2.calcHist([img], [0], None, [64], [0, 256])
    hist = hist.flatten()
    return hist / hist.sum() if hist.sum() != 0 else hist


def meanGradient(image):
    grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradxy


# 暗通道先验相关函数
def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx:]

    atmsum = np.zeros([1, 3])
    for ind in indices:
        atmsum = atmsum + imvec[ind]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * im + mean_b
    return q


# 特征提取相关函数
def to_dark_Channel(img, patch_size=15):
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(dc, kernel)
    return dark


def t_dark(img, omega=0.95, patch_size=15):
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    I = img.astype('float64') / 255.0
    dark = DarkChannel(I, patch_size)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, patch_size)

    # 使用导向滤波进行细化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float64') / 255.0
    te = Guidedfilter(gray, te, patch_size, 0.001)

    return np.clip(te, 0.1, 1.0)  # 限制透射率在 0.1 到 1 之间


def feature_extra(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确保输入图像是正确的格式
    if torch.is_tensor(img):
        img = img.cpu().numpy().transpose(1, 2, 0)

    # 将图像转换为 0-255 范围的 uint8 类型
    img2 = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    img2 = np.clip(img2, 0, 255).astype('uint8')
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 暗通道特征
    dark = to_dark_Channel(img2)
    fd = Histogram_cal(dark)
    fd = np.squeeze(fd)

    # 透射率特征
    te = t_dark(img2)
    te = (te * 255).astype('uint8')
    ft = Histogram_cal(te)
    ft = np.squeeze(ft)

    # 亮度特征
    fbr = cal_bright(img2)
    fbr = np.array([fbr])

    # 对比度特征
    fco = cal_contrast(img2)
    fco = np.array([fco])

    # 平均梯度特征
    gd_img = meanGradient(image)
    fmgd = Histogram_cal(gd_img)
    fmgd = np.squeeze(fmgd)

    # 归一化处理
    fd = fd / np.sum(fd) if np.sum(fd) != 0 else fd
    ft = ft / np.sum(ft) if np.sum(ft) != 0 else ft
    fmgd = fmgd / np.sum(fmgd) if np.sum(fmgd) != 0 else fmgd

    # 检查点：打印每个特征的统计信息
    # print(f"Dark channel histogram: min={fd.min():.4f}, max={fd.max():.4f}, mean={fd.mean():.4f}")
    # print(f"Transmission histogram: min={ft.min():.4f}, max={ft.max():.4f}, mean={ft.mean():.4f}")
    # print(f"Brightness: {fbr[0]:.4f}")
    # print(f"Contrast: {fco[0]:.4f}")
    # print(f"Mean gradient histogram: min={fmgd.min():.4f}, max={fmgd.max():.4f}, mean={fmgd.mean():.4f}")

    # 将特征转换为张量并拼接
    feature_tensor = torch.cat([
        torch.from_numpy(fmgd.astype(np.float32)),
        torch.from_numpy(fco.astype(np.float32)),
        torch.from_numpy(fbr.astype(np.float32)),
        torch.from_numpy(ft.astype(np.float32)),
        torch.from_numpy(fd.astype(np.float32))
    ], 0).to(device)

    return feature_tensor


# 使用示例
if __name__ == "__main__":
    # 假设我们有一个输入图像
    input_image = np.random.rand(3, 224, 224)  # 模拟一个随机的 3 通道图像
    input_tensor = torch.from_numpy(input_image).float()

    # 提取特征
    features = feature_extra(input_tensor)

    print("特征张量的形状:", features.shape)
    print("特征张量的设备:", features.device)