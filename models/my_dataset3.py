from PIL import Image
import torch
from torch.utils.data import Dataset
# import utilis1
from torchvision import transforms
import cv2
import numpy as np
from feature_fusion import feature_Extra3


#这个主要是为验证特征提取不同的特征准备，引用featureExtra不同
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, transform_TDD=None):
        self.images_path = images_path[0]
        self.images_path_TDD = images_path[1]
        self.images_class = images_class
        self.transform = transform
        self.transform_TDD = transform_TDD

    # 计算数据集下所有样本个数，注意传入的数据是哪种方式。
    def __len__(self):
        return len(self.images_path)

    # 每次传入索引返回该索引所对应图片和标签信息
    def __getitem__(self, item):
        # self.images_path[item]得到一个路径，item是一个索引，索引是我们的batch_size产生的。
        # 注意这里的图片是使用PIL处理的，如果需要opencv需要格式转换
        img = Image.open(self.images_path[item]).convert('RGB')
        img_area = Image.open(self.images_path_TDD[item]).convert('RGB')
        # print(self.images_path[item])
        # print(self.images_path_TDD[item])
        # print(img_area)
        label = self.images_class[item]
        # print("MyDataSet")
        #这部分是处理官方的数据，输出的图片已经是tensor形式
        if self.transform is not None:

            img = self.transform(img)
            # print(img.shape)

            img_TDD = self.transform(img_area)
            # print(img_TDD.shape)
        # if self.transform_TDD is not None:
        img_fog_area_tensor = feature_Extra3.feature_extra(img_TDD)

        #######
        # img1 = img.numpy().transpose(1, 2, 0)
        # # 反Normalize 与 归一化操作
        # img1 = (img1 * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        # img1 = Image.fromarray(np.uint8(img1))
        #
        # img1.show()
        # img2 = img_TDD.numpy().transpose(1, 2, 0)
        # # 反Normalize 与 归一化操作
        # img2 = (img2 * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        # img2 = Image.fromarray(np.uint8(img2))
        # img2.show()
        # dark_img = utilis1.to_dark_Channel(img2)
        # tensor_trans = transforms.ToTensor()
        # dark_img = tensor_trans(dark_img)
        # # torch.Size([1, 224, 224])
        # # print(dark_img.shape)
        ####

        return img, img_fog_area_tensor, label

    @staticmethod
    def collate_fn(batch):#打包方式，这里batch是外在输入，就是上面得到img和label构成的元组。如果batch_size为8，则此时batch是有8组数据，每组都是[图片tensor,label]
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, images_TDD, labels = tuple(zip(*batch))  # *batch非关键字参数，生成三组数据，每组有batch_size个

        images = torch.stack(images, dim=0)  # torch.stack拼接，增加一个维度，在0维度拼接（batch_size, c, w, h）
        images_TDD = torch.stack(images_TDD, dim=0)  # torch.stack拼接，增加一个维度，在0维度拼接（batch_size, c, w, h）
        labels = torch.as_tensor(labels)  # 将label转化为tensor
        return images, images_TDD, labels
