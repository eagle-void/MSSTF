import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import tensor
import torch.utils.data as Data
from torch.backends import cudnn
from einops import rearrange
import torch
import torch.nn as nn
import warnings
import random
import psutil
import os
import argparse
from model import MSSTF, get_model_name


############################################################################## 不知道干啥的但是好像不能删的设定
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机参数：保证实验结果可以重复
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # 适用于显卡训练
torch.cuda.manual_seed_all(SEED)  # 适用于多显卡训练
cudnn.benchmark = False
cudnn.deterministic = True

pause = psutil.Process(os.getpid())
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda")
############################################################################## 不知道干啥的但是好像不能删的设定


# 参数
def ParserBuild():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=32, help="img_size")
    parser.add_argument("--img_patch_size", type=int, default=32, help="img_patch_size")
    parser.add_argument("--piece", type=int, default=144, help="R_arrange")
    parser.add_argument("--input_dim", type=int, default=1, help="input_dim")
    parser.add_argument("--output_dim", type=int, default=1, help="output_dim")

    parser.add_argument("--win_size_long", type=int, default=72, help="window size")
    parser.add_argument("--win_size_short", type=int, default=6, help="window size")
    parser.add_argument("--batch", type=int, default=3, help="train batch size")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--p", type=float, default=3, help="Classification threshold")
    opt = parser.parse_args()
    return opt


# 数据类
class DatasetClass(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs.cpu())
        self.label = torch.FloatTensor(data_targets.cpu())

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


# 时序数据堆叠 x: b,t,d,h,w [t要在第二个维度]
def TimeSeriesStack(data, label):
    data_list = []
    label_list = []
    for i in range(data.shape[1] - opt.win_size_long + 1):
        data_list.append(data[:, i:i + opt.win_size_long])
        label_list.append(label[:, i:i + opt.win_size_long])
    data = rearrange(torch.stack(data_list), 'n b t d h w -> (n b) d t h w')
    label = rearrange(torch.stack(label_list), 'n b t d h w -> (n b) d t h w')

    batchnorm = torch.nn.BatchNorm3d(opt.input_dim)
    data = batchnorm(data)
    label = batchnorm(label)
    return data, label


# 增强数据集
def EnhancedDataset():
    # 读入数据
    data = torch.load('G:\\ProjectFile\\Dataset\\TrainDataset\\32\\Enhanced_FireData_Train.pt')
    label = torch.load('G:\\ProjectFile\\Dataset\\TrainDataset\\32\\Enhanced_FireLabel_Train.pt')

    # 随机划分训练集和测试集
    rand_idx = torch.randperm(data.shape[0])
    data = data[rand_idx]
    label = label[rand_idx]

    TrainData = data[:-2, :, 6:7] - data[:-2, :, 13:14]
    TrainTarget = label[:-2]
    TestData = data[-2:, :, 6:7] - data[-2:, :, 13:14]
    TestTarget = label[-2:]

    #TrainData = torch.cat((TrainData[:, :, 6:7], TrainData[:, :, 13:14], (TrainData[:, :, 6:7] - TrainData[:, :, 13:14])), dim=2)
    #TestData = torch.cat((TestData[:, :, 6:7], TestData[:, :, 13:14], (TestData[:, :, 6:7] - TestData[:, :, 13:14])), dim=2)

    TrainData, TrainTarget = TimeSeriesStack(TrainData, TrainTarget)
    TestData, TestTarget = TimeSeriesStack(TestData, TestTarget)

    train_dataset = DatasetClass(TrainData, TrainTarget)
    test_dataset = DatasetClass(TestData, TestTarget)
    TrainDataLoader = Data.DataLoader(train_dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
    TestDataLoader = Data.DataLoader(test_dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
    print('----------------------------------训练数据读入完成！----------------------------------')
    return TrainDataLoader, TestDataLoader


# 虚拟数据集
def SyntheticDataset():
    # 读入数据
    TrainData = torch.load('F:\\zlzzzlz\\Dataset\\TrainDataset\\32\\Synthetic_FireData_Train.pt')[:8]
    TrainTarget = torch.load('F:\\zlzzzlz\\Dataset\\TrainDataset\\32\\Synthetic_FireLabel_Train.pt')[:8]
    TestData = torch.load('F:\\zlzzzlz\\Dataset\\TrainDataset\\32\\Synthetic_FireData_Test.pt')[:2]
    TestTarget = torch.load('F:\\zlzzzlz\\Dataset\\TrainDataset\\32\\Synthetic_FireLabel_Test.pt')[:2]

    TrainData, TrainTarget = TimeSeriesStack(TrainData, TrainTarget)
    TestData, TestTarget = TimeSeriesStack(TestData, TestTarget)

    train_dataset = DatasetClass(TrainData, TrainTarget)
    test_dataset = DatasetClass(TestData, TestTarget)
    TrainDataLoader = Data.DataLoader(train_dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
    TestDataLoader = Data.DataLoader(test_dataset, batch_size=opt.batch, shuffle=True, drop_last=True)
    print('----------------------------------训练数据读入完成！----------------------------------')
    return TrainDataLoader, TestDataLoader


# 模型训练
def Train(traindataloder, testdataloder):
    # 模型加载
    model = MSSTF(win_size=opt.win_size_short, input_dim=opt.input_dim)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)  # 优化器
    optimizer = torch.nn.DataParallel(optimizer, device_ids=[0, 1, 2]).module
    criterion = nn.MSELoss().cuda()  # 损失函数
    #criterion = nn.CrossEntropyLoss().cuda()

    # 模型训练
    model_name = get_model_name()
    os.makedirs('result', exist_ok=True)
    val_loss = []
    train_loss = []
    best_test_loss = 10000000
    best_model = model
    for epoch in range(opt.epochs):
        print('\n' + '----------------------------------第' + str(epoch+1) + '轮模型训练开始----------------------------------')
        train_epoch_loss = []
        for index, (inputs, targets) in tqdm(enumerate(traindataloder), desc="第{}轮梯度下降".format(epoch+1)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            # loss
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
            train_epoch_loss.append(loss.item())
            torch.cuda.empty_cache()
            #print("epoch:", epoch + 1, "\t  train_epoch_loss:", round(loss.item(), 6))

        train_loss.append(np.mean(train_epoch_loss))
        val_epoch_loss = Examine(model, testdataloder, criterion)  # 模型测试
        val_loss.append(val_epoch_loss)
        print("epoch:", epoch + 1,
              "\t  train_epoch_loss:", round(np.mean(train_epoch_loss), 6),
              "\t  val_epoch_loss:", round(val_epoch_loss, 6))

        # 保存下来最好的模型：
        if val_loss[-1] < best_test_loss:
            best_test_loss = val_loss[-1]
            best_model = model
            torch.save(best_model.state_dict(), 'result\\' + model_name + '_BestTrainModel.pth')
        # torch.save(model.state_dict(), 'result\\STTransformer_LastTrainModel.pth')
    print("best_test_loss ----------------------------------------------", round(best_test_loss, 6))  # 输出最优模型的训练损失
    print("The best model has been saved!")
    LossVisible(train_loss, val_loss, model_name)  # 误差损失可视化


# 测试函数
def Examine(model, testdataloder, criterion):
    val_epoch_loss = []
    with torch.no_grad():
        for index, (inputs, targets) in tqdm(enumerate(testdataloder)):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            # loss
            loss = criterion(outputs, targets)
            val_epoch_loss.append(loss.item())

    return np.mean(val_epoch_loss)


# 训练及测试损失可视化函数
def LossVisible(x, y, model_name):
    fig = plt.figure(facecolor='white', figsize=(10, 7))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xmax=len(y), xmin=0)
    plt.ylim(ymax=max(max(x), max(y)), ymin=min(min(x), min(y)))
    # 画两条（0-9）的坐标轴并设置轴标签x，y
    x1 = [i for i in range(0, len(y), 1)]  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
    y1 = y  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
    x2 = [i for i in range(0, len(x), 1)]
    y2 = x
    colors1 = '#00CED4'  # 点的颜色
    colors2 = '#DC143C'
    # 画折线图
    plt.plot(x1, y1, c=colors1, alpha=0.4, label='val_loss')
    plt.plot(x2, y2, c=colors2, alpha=0.4, label='train_loss')
    plt.legend()

    # save figure
    plt.savefig('result\\' + model_name + '_TrainLossVisible.jpg')
    plt.show()


############################################################################## 计算主程序
if __name__ == '__main__':
    # 网络参数
    opt = ParserBuild()

    # 执行训练
    TrainDataLoader, TestDataLoader = SyntheticDataset()  # 虚拟数据集
    #TrainDataLoader, TestDataLoader = EnhancedDataset()  # 增强数据集
    Train(TrainDataLoader, TestDataLoader)
