# 复现 hu 的分成四个集合的方法
# 
# 1、网络：cnnp
# 2、输入：原始512大小灰度图
# 3、处理：用其他三个集合去预测一个集合
# 4、输出：510大小的预测图像
# 6、学习率：不变，一直保持同一个
# 7、训练轮数：120
# 8、batch大小：1




import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import os
import time
import random
import csv
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 导入网络结构
from cnnp import *

# batch1:单个集合的点512*512
# raw1:另一个集合的点510*510，在外面先用原图像乘过mask1了，然后再缩小一圈得到
# mask1:集合的掩码
# mask_c:
def trainOneBatch(batch1, target, mask1):
    # 进行预测，输出一个510*510的图
    mycnn.train()
    with torch.enable_grad():
        output = mycnn(batch1)
    # 使用一个掩码与output进行计算，得到
    output = output*mask1 # 广播，得到1/4集合
    loss = loss_function(output, target)  # 计算损失,只学习复杂部分的损失

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 再开始反向传播，不然会累计上一次的
    optimizer.step()  # 更新参数
    # 输出loss
    return loss

# 读取图片，后面需要多次调用这个函数
def get_image(file_path1):
    img = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)  # cv2打开的图片时RGB三通道的，处理一下为单通道灰度图
    img = img.astype('float64')
    return img


# 训练时使用的参数

# 超参数
LR = 0.001  # 学习率
BATCH_SIZE = 8 # 一个batch的大小，要对每个图像进行自适应的调整
Weight_Decay = 0  # L2正则化项参数
EPOCHES = 10000  # 训练轮数
# 保存的图片，模型命名
name = "epoch_"
path_num = 4
# x1_lr001_bs4_ep30_wt00001
begin_epoch = 0


# 载入的模型路径名
# name_model = "train_encoder_x1_adam_cnn_encoder5_2"  # 调用学习好的继续训练

# 获取gpu是不是可用
cuda_available = torch.cuda.is_available()  # 如果可用需要把模型和数据都挂在GPU上

# 实例化网络，这个网络可以随意改
print('===开始训练===')
mycnn = CNNP()
# mycnn_path = "/home/yangx263/CNN_RDH/models/model_imageNet_cubic/model_my_adam_L1_all/" + name + str(begin_epoch) + ".pkl"
#checkpoint = torch.load(mycnn_path)
#mycnn.load_state_dict(checkpoint['network'])

# 复制模型参数
# 当用到验证集的时候可以考虑，使用验证集的数据来进行择优，将最好的参数复制下来进行下一次的学习
best_model_wts = copy.deepcopy(mycnn.state_dict())
best_acc = 0.0  # 记录最好的正确率
best_loss = 10000000.0  # 记录最好的loss，感觉也可以不记录

# 画图用到的
train_loss_all = []  # 记录所有轮次的loss(每轮的平均)
train_acc_all = []
val_loss_all = []
val_acc_all = []

since = time.time()  # 开始的时间

if cuda_available:
    mycnn.cuda()  # 这里就要先挂在GPU上，因为下面定义优化器的时候要用上模型

# 定义优化器和损失函数
# 优化器，等会放到循环里，修改lr
# optimizer = torch.optim.SGD(mycnn.parameters(), lr=LR, weight_decay=Weight_Decay)  # 定义优化器，要用上模型

optimizer = torch.optim.AdamW(mycnn.parameters(), lr=LR, weight_decay=0.001)  # 定义优化器，要用上模型，为啥用这个模型会有问题呢
#optimizer.load_state_dict(checkpoint['optimizer'])
loss_function = nn.MSELoss()  # 定义损失函数，这里使用均方误差，归一化问题，主要是预测的越准确越好

# 处理训练集的数据，输入原始图像
imgs_train = []  # 构造一个存放图片的列表数据结构
imgs_train_target = []
imgs_val = []  # 构造一个存放图片的列表数据结构
imgs_val_target = []


# 只存原图像的地址
# 读取图片
files_train_path = []
DIRECTORY_data = "/home/yangx/CNN_RDH/spl/3000/"  # 这里是自己的图片的位置
files_data = os.listdir(DIRECTORY_data)  # 将文件名读取到列表中
files_data = sorted(files_data)
for file in files_data:
    file_path = DIRECTORY_data + "/" + file  # 文件的路径
    files_train_path.append(file_path)


files_test_path = []
test_path = '/home/yangx/CNN_RDH/spl/test32/'
test_data = os.listdir(test_path)  # 将文件名读取到列表中
test_data = sorted(test_data)
for file in test_data:
    file_path = test_path + "/" + file  # 文件的路径
    files_test_path.append(file_path)

# block_size = 128

# 生成一个掩码
mask_1000 = [] # 大尺寸掩码
img_ans = np.zeros((1000, 1000))
for i in range(0, 1000, 2):
    for j in range(0, 1000, 2):
        img_ans[i, j] = 1
mask_1000.append(img_ans)
Mask_1000 = torch.FloatTensor(mask_1000).unsqueeze(dim=3).permute(0, 3, 1, 2)
if cuda_available:
    Mask_1000 = Mask_1000.cuda()
# prediction kernel
mykernel = [[0,1,0,1,0],[1,2,4,2,1],[0,4,0,4,0],[1,2,4,2,1],[0,1,0,1,0]]
mykernel = np.array(mykernel)
mykernel = mykernel / np.sum(mykernel)


# 开始训练
print("train")
# 遍历迭代期
optimizer.param_groups[0]['lr'] = 0.001 * 0.95 ** (begin_epoch // 15)
for epoch in range(begin_epoch+1,EPOCHES):
    print('-' * 10)  # 打印10个----
    print('Epoch {}/{} shuffle{}'.format(epoch, EPOCHES - 1,len(files_train_path)))  # 因为下标从0开始，所以计数也从0开始
    if epoch % 15 == 0:
        optimizer.param_groups[0]['lr'] = 0.001 * 0.95 ** (epoch // 15)
    print('lr:',optimizer.param_groups[0]['lr'])

    # 每个epoch有两个训练阶段
    train_loss = 0.0
    train_corrects = 0  # 这里是回归问题，就没有精度这个问题了
    train_num = 0

    # np.random.seed(0)
    np.random.shuffle(files_train_path)

    count = 0  # count是为了凑齐成为一个batch_size的大小
    batch = []
    y = []
    pred = []

    block_size = 128#np.random.randint(128,256+1)
    # block_size = 64
    print('block_size:',block_size)
    for j in range(len(files_train_path)):  # 遍历完了所有的数据
        # 读图片,先不着急拆成两个
        file_path = files_train_path[j]
        img = get_image(file_path)
        img = np.rot90(img, k=np.random.randint(4))
        x_position = np.random.randint(512-block_size+1)
        y_position = np.random.randint(512-block_size+1)
        img = img[x_position:x_position+block_size,y_position:y_position+block_size]
        # target = img[2:510, 2:510]
        # target = img[2:-2,2:-2]
        target = img[:,:]

        img += np.random.randint(-1, 2, (block_size, block_size))
        pre = cv2.filter2D(img, -1, mykernel)

        batch.append(img)
        pred.append(pre)
        y.append(target)

        count += 1

        if count == BATCH_SIZE or j == len(files_train_path) - 1:  # 这里就算最后
            # 当达到一个batch大小的时候
            # 列表转成张量，再转换维度

            batch_img = torch.FloatTensor(batch).unsqueeze(dim=3).permute(0, 3, 1, 2) / 255
            pred_img = torch.FloatTensor(pred).unsqueeze(dim=3).permute(0, 3, 1, 2) / 255
            y_img =  torch.FloatTensor(y).unsqueeze(dim=3).permute(0, 3, 1, 2) / 255

            # 张量
            if cuda_available:
                batch_img = batch_img.cuda()  # 数据变换到gpu上
                pred_img = pred_img.cuda()
                y_img = y_img.cuda()

            Mask_big = Mask_1000[:,:,0:block_size,0:block_size]
            Mask_small = Mask_1000[:,:,2:block_size-2,2:block_size-2]
            input_batch = batch_img - batch_img * Mask_big + pred_img * Mask_big
            # target_batch = y_img - pred_img[:,:,2:block_size-2,2:block_size-2]  # 在前面已经缩小了一圈
            # target_batch = target_batch * Mask_small
            target_batch = y_img - pred_img
            target_batch = target_batch * Mask_big
            # 训练，更新参数
            loss1 = trainOneBatch(input_batch, target_batch, Mask_big)  # 训练一个批次，一个批次的平均loss
            train_loss += loss1.item() * batch_img.size(0)  # 这个批次的总loss，平均×数量
            train_num += batch_img.size(0)  # 这个批次的总数

            batch = []
            pred = []
            y = []
            count = 0
    # 计算在一个epoch在训练集的损失
    train_loss_all.append(train_loss / train_num)
    print('{} Train Loss: {:.8f}'.format(epoch, train_loss_all[-1]))
    with open('/home/yangx/CNN_RDH/spl/trainloss.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_to_write = [epoch]+['{:.8f}'.format(train_loss_all[-1])]
        writer.writerow(row_to_write)
    time_use = time.time() - since
    print('{} Train complete in {:.0f}m{:.0f}s'.format(epoch, time_use // 60, time_use % 60))  # 训练花费的时间

    # test loss
    test_loss = []
    # np.random.shuffle(files_test_path)
    test_count = 0
    for j in range(len(files_test_path)):
        if test_count == 100:
            break
        test_count += 1
        file_path = files_test_path[j]
        img = get_image(file_path)
        img = np.rot90(img, k=np.random.randint(4))
        m,n = img.shape[0],img.shape[1]
        pre = cv2.filter2D(img, -1, mykernel)
        # target = img[2:-2,2:-2]
        target = img[:,:]
        batch.append(img)
        pred.append(pre)
        y.append(target)
        batch_img = torch.FloatTensor(batch).unsqueeze(dim=3).permute(0, 3, 1, 2) / 255
        pred_img = torch.FloatTensor(pred).unsqueeze(dim=3).permute(0, 3, 1, 2) / 255
        y_img = torch.FloatTensor(y).unsqueeze(dim=3).permute(0, 3, 1, 2) / 255
        if cuda_available:
            batch_img = batch_img.cuda()  # 数据变换到gpu上
            pred_img = pred_img.cuda()
            y_img = y_img.cuda()
        Mask_big = Mask_1000[:, :, 0:m, 0:n]
        Mask_small = Mask_1000[:, :, 2:m-2, 2:n-2]
        input_batch = batch_img - batch_img * Mask_big + pred_img * Mask_big
        target_batch = y_img - pred_img#[:, :, 2:m-2, 2:n-2]
        target_batch = target_batch * Mask_big
        mycnn.eval()
        with torch.no_grad():
            output = mycnn(input_batch)
            # 使用一个掩码与output进行计算，得到
            output = output * Mask_big  # 广播，得到1/4集合
            loss = loss_function(output, target_batch)  # 计算损失,只学习复杂部分的损失
        test_loss.append(loss.item())
        batch = []
        pred = []
        y = []
    test_loss_aver = sum(test_loss) / len(test_loss)
    with open('/home/yangx/CNN_RDH/spl/testloss.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        row_to_write =  [epoch]+['{:.8f}'.format(test_loss_aver)]
        writer.writerow(row_to_write)
    print('{} Test Loss: {:.8f}'.format(epoch, test_loss_aver))
    time_use = time.time() - since
    print('{} Test complete in {:.0f}m{:.0f}s'.format(epoch, time_use // 60, time_use % 60))

    # 每30轮保存一次模型
    if True:
        mycnn_path = "/home/yangx/CNN_RDH/spl/models/" + name + str(epoch) + ".pkl"  # 文件的路径
        # torch.save(mycnn, mycnn_path)

        state = {'network': mycnn.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, mycnn_path)

        # mycnn_param_path = "/home/yangx263/CNN_RDH/models/model_newloss508_v2/" + name + str(i+1) + "_param.pkl"
        # torch.save(mycnn.state_dict(), mycnn_param_path)

print('\n---------over----------\n')
