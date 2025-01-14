import os
import sys
import json
import pickle
import random
import numpy as np

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data_two(root: str, rootNew: str, val_rate: float = 0.3):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    assert os.path.exists(rootNew), "dataset root: {} does not exist.".format(rootNew)

    # 遍历文件夹，一个文件夹对应一个类别。
    # os.listdir(root)；列出所有文件， os.path.isdir(os.path.join(root, cla))判断os.path.join(root, cla)是否是文件夹，是则该文件夹名字保存到flower_class
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # flower_class：[每个文件夹的名字]

    # 排序，保证顺序一致
    flower_class.sort()

    # 生成类别名称以及对应的数字索引
    # （key,val） = (类别名称，索引)
    class_indices = dict((k, v) for v, k in enumerate(flower_class))

    # dict((val, key) for key, val in class_indices.items()，key,val值反过来，现在字典键 = 索引，值=类别
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # 写入json
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_path_new = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息

    val_images_path = []  # 存储验证集的所有图片路径
    val_images_path_new = []
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件flower_class存储的是类别名称，也就是类别文件夹名称
    for cla in flower_class:
        # 根目录核+类别名称，拼成一个文件夹完整路径
        cla_path = os.path.join(root, cla)
        cla_path_new = os.path.join(rootNew, cla)

        # 遍历获取supported支持的所有文件路径
        # 1、for i in os.listdir(cla_path)：遍历文件夹中文件
        # 2、 if os.path.splitext(i)[-1]，对文件名字i进行分割，得两个元素，名字和后缀。取后缀判断是否在我们支持得后缀文件里面
        # 3、将文件夹中我们支持得后缀图片文件，拼接成一个完整得路径。放到images中
        # images是一个存储一个个图片文件路径得路径列表，这里需要修改，改成两个文件列表归为一类。
        # root 根目录，cla 类别名称， i图片文件名称
        images = []
        root_pic_name = os.listdir(cla_path)
        rootNew_pic_name = os.listdir(cla_path_new)
        sky_name = ''
        for i in root_pic_name:
            # 确保两个文件夹都存在该图
            if i[1] == '-':
                DDTIMG_name = i.split("-")[-1]
            else:
                DDTIMG_name = i
                # 解决大小写.JPG无法读取的问题
                # DDTIMG_name1 = i.split(".")[0]
                DDTIMG_name1 = i.rsplit(".", 1)[0]

            if ((os.path.splitext(DDTIMG_name)[-1]) in supported) and ((DDTIMG_name1 + '.jpg' in rootNew_pic_name) or
                                                                       (DDTIMG_name1 + '.JPG' in rootNew_pic_name)):
                images.append(os.path.join(root, cla, i))
            else:
                print(f"{cla_path_new}中找不到{i}")

        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本， k是选取个数，放测试集列表
        val_path = random.sample(images, k=int(len(images) * val_rate))  # 样本中含扩展数
        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)

                imgName = img_path.split("\\")[-1]
                if imgName[1] == "-":
                    imgName = imgName.split('-')[-1]
                img_path_new = os.path.join(rootNew, cla, imgName)
                val_images_path_new.append(img_path_new)

            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

                imgName = img_path.split("\\")[-1]
                if imgName[1] == "-":
                    imgName = imgName.split('-')[-1]
                img_path_new = os.path.join(rootNew, cla, imgName)
                train_images_path_new.append(img_path_new)

    train_images_two_path = [train_images_path, train_images_path_new]
    val_images_two_path = [val_images_path, val_images_path_new]

    # val_images_path_two：[原图列表，天空列表]  ，train_images_two_path1：[[原图， 天空图]，。。。， [原图，天空图]]

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    # 这个是绘制训练图像数量直方图
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_two_path, train_images_label, val_images_two_path, val_images_label


def read_split_data(root: str, val_rate: float = 0.3):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('../class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", '.jpeg']  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = '../class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, images1, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device), images1.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad() # 在接下来过程中，不要计算每个节点的误差损失梯度，如果不用这个函数，则在测试过程中也会计算损失误差梯度，它会消耗算力和内存资源。
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, images1, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device), images1.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def misclassified_images(pred_y, writer, target, images, label, count, num, output=""):
    misclassified = (pred_y != target.data)  # 判断是否一致,输出一个true和false的列表
    # print(images[misclassified].shape[0])
    index_err = [i for i, x in enumerate(misclassified) if x]
    # print(index_err)
    ct = 0

    for err_index in index_err:
        image_tensor = images[err_index]
        print(err_index)
        # print(image_tensor.shape)
        img = image_tensor.cpu().numpy().transpose(1, 2, 0)
        # resnet
        img = ((img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype('uint8')

        # vgg
        # img = ((img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255).astype('uint8')
        # print(img.shape)
        img_name = 'Predict-{}__Actual-{}'.format(label[pred_y.tolist()[err_index]],
                                                              label[target.tolist()[err_index]])
        # print(img_name)
        image_name = "epoch_"+str(num)+"_"+str(index_err[ct])+".jpg"

        # print(save)
        imgfile = '{root}/{file_name}'.format(root=output, file_name = img_name)
        if os.path.exists(imgfile) is False:
            os.makedirs(imgfile)

        save = '{root}/{file_name}/{img}'.format(root=output, file_name=img_name, img=image_name)
        plt.imshow(img)
        plt.savefig(save)
        writer.add_image(img_name, img, num, dataformats='HWC')
        ct += 1


def data_save(root, file):
    if not os.path.exists(root):
        with open(root, 'w') as _:
            pass
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()

    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()


def draw_picture(path):
    # 画训练结果图
    train_acc = []
    train_loss = []
    val_acc = []
    for name in os.listdir(path):
        imfile = os.path.join(path, name)
        # print(imfile)
        file_temp = open(imfile, 'r')
        lines = file_temp.readlines()

        if name == "train_acc.txt":
            for line in lines:
                tmp = line.split(" ")[1]
                train_acc.append(float(tmp.split("\\")[0]))

        if name == "val_acc.txt":
            for line in lines:
                tmp = line.split(" ")[1]
                val_acc.append(float(tmp.split("\\")[0]))
        epoch = lines[-1][:lines[-1].index(' ')]
        file_temp.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(int(epoch) + 1)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # linestyle = "solid"  "dashed" "dashdot" "dotted"; linewidth=3.0
    ax.plot(x, train_acc, color="red", label="train_acc")  # 用名字代表颜色, 'red'代表红色
    ax.plot(x, val_acc, color="b", label="val_acc")  # 颜色代码，(rgbcmyk)

    ax.set_title("结果对比", fontdict={"fontsize": 15})
    ax.set_xlabel("轮数")  # 添加横轴标签
    ax.set_ylabel("准确率")  # 添加纵轴标签
    ax.legend(loc="best")  # 展示图例

    plt.show()


def image_show(inp):
    """
    inp:图像的tensor（b,c,x,h）
    显示tensor图片
    """
    plt.figure(figsize=(14, 3))
    # 变为numpy
    inp = inp.numpy().transpose((1, 2, 0))
    # 逆归一化 std 标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # unnormalize

    # inp = np.clip(inp, 0, 1)#值限制在01之间
    plt.pause(0.001)

    plt.imshow(inp.astype('uint8'))  # 图片需要转换为int8类型
