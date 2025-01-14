import os
import argparse
import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset3 import MyDataSet
from utils import read_split_data, read_split_data_two, train_one_epoch, evaluate, data_save, draw_picture
from vit_model import vit_base_patch16_224_in21k as create_model

def main(args, CNN=None):
    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)

    if args.logs:
        if os.path.exists(args.tensorboard_path) is False:
            os.makedirs(args.tensorboard_path)

        if not os.path.exists(args.train_log_txt):
            os.makedirs(args.train_log_txt)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("../weights") is False:
        os.makedirs("../weights")

    tb_writer = SummaryWriter(log_dir=args.tensorboard_path)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data_two(args.data_path, args.data_CNN_path)
    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=0,
                                               )
                                               # collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=0,
                                             )
                                             # collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    if args.tran_learn:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

    # if args.freeze_layers:
    #     print("=================")
    #     for name, para in model.named_parameters():
    #         # 除head外，其他权重全部冻结
    #         if "head" not in name and "layers.3" not in name and "layers.2" not in name:
    #         # if "head" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    best_acc = 0.0
    for epoch in range(args.epochs):
        start_t = datetime.datetime.now()
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        end_t = datetime.datetime.now()
        print("运行一次时间", (end_t - start_t).seconds)
        if args.logs:

            train_loss_tmp = {}
            train_acc_tmp = {}
            val_acc_tmp = {}
            val_loss_tmp = {}
            train_loss_tmp[epoch] = round(train_loss, 4)
            train_acc_tmp[epoch] = round(train_acc, 4)
            val_acc_tmp[epoch] = round(val_acc, 4)
            val_loss_tmp[epoch] = round(val_loss, 4)
            data_save(args.train_log_txt + '/train_loss.txt', train_loss_tmp)
            data_save(args.train_log_txt + '/train_acc.txt', train_acc_tmp)
            data_save(args.train_log_txt + '/val_acc.txt', val_acc_tmp)
            data_save(args.train_log_txt + '/val_loss.txt', val_loss_tmp)

            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        print(f"best_acc:{round(best_acc, 3)}", f"log_time : {args.log_time}")
        accurate = val_acc
        if accurate > best_acc:
            best_acc = accurate
            print(
                f"save model epoch:{epoch},train_accurate:{round(train_acc, 3)}, val_accurate:{round(val_acc, 3)}")
            torch.save(model.state_dict(), args.save_path + "/model-{}.pth".format(epoch))
    if args.logs:
        draw_picture(args.train_log_txt)


if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lfr', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--cutmix_alpha', type=float, default=0.5)

    # 数据集所在根目录
    # # http://download.tensorflow.org/example_images/flower_photos.tgz
    path1 = "/data/DL/code/visibility/haze_simu2"
    path2 = "/data/DL/code/visibility/haze_simu2_region"
    parser.add_argument('--data-path', type=str,
                        default=path1)
    parser.add_argument('--data-CNN-path', type=str,
                        default=path2)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/data/DL/code/visibility/resnet/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 保存训练准确率损失结果txt
    parser.add_argument('--train_log_txt', type=str, default="../log/log_txt/swin_small_patch4_window7_224/" + current_time,
                        help='path of train_data')
    parser.add_argument('--log_time', type=str, default=current_time,
                        help='log_time')
    # 保存最好权重路径
    parser.add_argument('--save_path', type=str, default="../weights/after_train/swin_small_patch4_window7_224/" + current_time,
                        help='path of sava weights')
    # tensorboard
    parser.add_argument('--tensorboard_path', type=str, default="../tensorboard/swin_small_patch4_window7_224/" + current_time,
                        help='path of sava weights')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--tran_learn', type=bool, default=True)
    parser.add_argument('--logs', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
