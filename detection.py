import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from torch.backends import cudnn
import torch
import warnings
import random
import psutil
import os
from model import MSSTF, get_model_name
from train import ParserBuild

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda:0")
############################################################################## 不知道干啥的但是好像不能删的设定


# 可燃物掩膜计算
def NDVI_mask(data):
    ndvi = (data[:, 3] - data[:, 2]) / (data[:, 3] + data[:, 2] + 1e-8)
    ndvi, _ = torch.max(ndvi, dim=0)
    ndvi = torch.where((ndvi > 0.23), torch.ones_like(ndvi), torch.zeros_like(ndvi))
    return ndvi


# 水掩膜计算
def Water_mask(data):
    data_daytime = torch.cat((data[:60, 5], data[132:, 5]), dim=0)
    water = torch.where((data_daytime > 0.05), torch.ones_like(data_daytime), torch.zeros_like(data_daytime))
    water = torch.sum(water, dim=0)
    water = torch.where((water > 60), torch.ones_like(water), torch.zeros_like(water))
    return water


# 云掩膜计算
def Cloud_mask(data):
    j1_d1 = (data[:60, 2] + data[:60, 3] > 1.2) & (data[:60, 14] < 265)
    j2_d1 = (data[:60, 2] + data[:60, 3] > 0.7) & (data[:60, 14] < 285)
    j1_d2 = (data[132:, 2] + data[132:, 3] > 1.2) & (data[132:, 14] < 265)
    j2_d2 = (data[132:, 2] + data[132:, 3] > 0.7) & (data[132:, 14] < 285)
    j_n = (data[60:132, 6] < 285) & (data[60:132, 14] < 265)

    cloud_d1 = torch.where((j1_d1 | j2_d1), torch.ones_like(data[:60, 2]), torch.zeros_like(data[:60, 2]))
    cloud_d2 = torch.where((j1_d2 | j2_d2), torch.ones_like(data[132:, 2]), torch.zeros_like(data[132:, 2]))
    cloud_n = torch.where(j_n, torch.ones_like(data[60:132, 6]), torch.zeros_like(data[60:132, 6]))

    cloud = torch.cat((cloud_d1, cloud_n, cloud_d2), dim=0)
    return cloud


# 掩膜计算整合
def Mask_process(data):
    ndvi = NDVI_mask(data)
    water = Water_mask(data)
    cloud = Cloud_mask(data)
    return ndvi, water, cloud


# 生成时间列表
def TimeList():
    # 时间列表
    time_list = []
    for h in range(24):
        for m in range(6):
            if h < 10:
                h_str = "0" + str(h)
            else:
                h_str = str(h)
            if m == 0:
                m_str = str("00")
            else:
                m_str = str(m * 10)

            time_list.append(h_str + ":" + m_str)
    return time_list


# 时序数据堆叠
def TimeSeriesStack(data):
    data_list = []
    for i in range(data.shape[0] - opt.win_size_long + 1):
        data_list.append(data[i:i + opt.win_size_long])

    batchnorm = torch.nn.BatchNorm3d(opt.input_dim)
    data = batchnorm(rearrange(torch.stack(data_list), 'b t d h w -> b d t h w'))
    return data


# 执行检测
def FireDetection(model, data, ndvi, water, cloud, site, resultoutput=True):
    piece, _, L, R = data.shape
    # data reshape
    data = TimeSeriesStack(data).cuda()

    # (t,l,r) 分类结果
    fire_pred = torch.zeros((piece, L, R)).cuda()
    output_list = []
    with torch.no_grad():
        for i in range(data.shape[0]):
            output_list.append(model(data[i:i+1])[0, 0])

    for t in range(piece):
        if t < opt.win_size_long:
            fire_pred[t] = output_list[0][t]
        else:
            fire_pred[t] = output_list[t - opt.win_size_long + 1][-1]

    # 结果输出
    if resultoutput:
        torch.save(fire_pred, 'result\\' + site + '_fire_pred.pt')

    # 结果整合
    fire_condition = (fire_pred > opt.p) & (water.cuda() == 0) & (ndvi.cuda() == 1) & (cloud.cuda() == 0)
    fire_pred = torch.where(fire_condition, torch.ones_like(fire_pred), torch.zeros_like(fire_pred))
    fire_pixel = torch.sum(fire_pred, dim=0)

    return fire_pred, fire_pixel


# 火点分类结果可视化
def ShowImage(firedata, img_lon, img_lat, site):
    # 创建一个画布
    fig, ax = plt.subplots(figsize=(11, 7))  # 设置画布大小
    im = ax.imshow(firedata.clone().detach().cpu().numpy(), cmap='Reds', aspect='auto', vmax=144, vmin=0)  # 设置热力图颜色与热力块大小
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # create colorbar色条
    cbar = ax.figure.colorbar(im, ax=ax)

    # color色条本身上刻度值大小及字体设置
    cbar.ax.tick_params(labelsize=9)
    cbarlabels = cbar.ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in cbarlabels]

    # 热力图设置
    ax.set_xticks(np.arange(img_lat))
    ax.set_yticks(np.arange(img_lon))

    # 用各自的列表项来标记他们
    tick_x = [0] * img_lat
    tick_y = [0] * img_lon
    for i in range(img_lon):
        tick_y[i] = i + 1
    for j in range(img_lat):
        tick_x[j] = j + 1
    ax.set_xticklabels(tick_x)
    ax.set_yticklabels(tick_y)

    # 关闭网格并用白色加宽网格
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(img_lat + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(img_lon + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 添加每个热力块的具体数值
    #for i in range(img_lon):
    #    for j in range(img_lat):
    #        text = ax.text(j, i, round(firedata[i][j], 2), ha="center", va="center", color="k", fontsize=6, fontname='Times New Roman')

    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=8)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置横纵坐标的名称及热力图名称以及对应字体格式
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 24,
             }

    # 横纵轴的名称
    #plt.xlabel('Row', font1)
    #plt.ylabel('Line', font1)

    # 热力图名称
    ax.set_title(site, font1)

    # 图的输出
    plt.savefig('result\\' + site + '_' + model_name + '_FireResult.jpg')
    #plt.show()


# 检测时间信息输出
def Temporal_Info(data_pre, data_ture, site, fire_list):
    for l in range(opt.img_size):
        for r in range(opt.img_size):
            # 起火信息输出
            fire_temp = []
            for t in range(opt.piece):
                if len(fire_temp) == 0 and data_ture[t, l, r] == 1.0:
                    fire_temp.append('fire')
                    if torch.sum(data_pre[t:t+4, l, r]) >= 1:
                        fire_info = site + ',' + str(l) + ',' + str(r) + ',' + str(t) + ',' + str(1) + '\n'
                    else:
                        fire_info = site + ',' + str(l) + ',' + str(r) + ',' + str(t) + ',' + str(0) + '\n'
                    fire_list.append(fire_info)
    return fire_list


# 精度指标计算
def AccuracyCount(data_pre, data_ture, site):
    data_ture = data_ture.cuda()

    # 精度计算
    TP, FN, FP, TN = 0, 0, 0, 0
    TP = ((data_ture == 1) & (data_pre == 1)).sum().item()
    FN = ((data_ture == 1) & (data_pre == 0)).sum().item()
    FP = ((data_ture == 0) & (data_pre == 1)).sum().item()
    TN = ((data_ture == 0) & (data_pre == 0)).sum().item()

    bias = 1e-7
    F1 = 2 * TP / (2 * TP + FP + FN + bias)
    IOU = TP / (FP + TP + FN + bias)
    Precision = TP / (TP + FP + bias)
    Recall = TP / (TP + FN + bias)
    PrecisionIndex = site + ":  F1 = %1.4f, IOU = %1.4f, Precision = %1.4f, Recall = %1.4f" % (F1, IOU, Precision, Recall) + '\r'
    with open(result_filename, 'a') as f:
        f.write(PrecisionIndex)
    print(PrecisionIndex)
    return [F1, IOU, Precision, Recall]


# 精度指标计算
def AccuracyCountAll(fire_list, precision_list):
    # temporal accuracy
    count = 0.0
    firenum = len(fire_list)
    for i in range(firenum):
        fireline = fire_list[i].strip().split(',')
        count += float(fireline[-1])
    accuracy = count / firenum
    print('Temporal Accuracy:' + str(accuracy))

    # average precision
    precision_list = np.asarray(precision_list)
    average_precision = np.mean(precision_list, axis=0)
    print("Average_precision:  F1 = %1.4f, IOU = %1.4f, Precision = %1.4f, Recall = %1.4f" %
          (average_precision[0], average_precision[1], average_precision[2], average_precision[3]) + '\r')

    # 结果输出
    with open(result_filename, 'a') as f:
        f.write("Average_precision:  F1 = %1.4f, IOU = %1.4f, Precision = %1.4f, Recall = %1.4f" %
                (average_precision[0], average_precision[1], average_precision[2], average_precision[3]) + '\r')
        f.write('Temporal accuracy:  ' + str(accuracy) + '\r')


# 单一火点事件探查
def SingleFireEventDetection():
    fire_list = []
    precision_list = []
    for site in site_list:
        # 数据加载【缺少数据去噪】
        data = torch.load('G:\\ProjectFile\\Dataset\\FireData\\32\\' + site + 'FireData.pt')
        label = torch.load('G:\\ProjectFile\\Dataset\\FireData\\32\\' + site + 'FireLabel.pt')
        ndvi, water, cloud = Mask_process(data)

        # 加载训练好的模型
        model = MSSTF(win_size=opt.win_size_short, input_dim=opt.input_dim)
        model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
        model.load_state_dict(torch.load('result\\' + model_name + '_BestTrainModel.pth'))
        model.eval()

        # 执行检测
        dataset = data[:, 6:7] - data[:, 13:14]
        fire_pred, fire_pixel = FireDetection(model, dataset, ndvi, water, cloud, site)

        # 检测结果可视化
        print(site + ' Fire Detection Finished!')
        precision_list.append(AccuracyCount(fire_pred, label, site))
        ShowImage(fire_pixel, opt.img_size, opt.img_size, site)
        fire_list = Temporal_Info(fire_pred, label, site, fire_list)

    print('Early Stage Fire Detection Result:')
    AccuracyCountAll(fire_list, precision_list)


############################################################################## 计算主程序
if __name__ == '__main__':
    # 网络参数
    opt = ParserBuild()
    model_name = get_model_name()
    result_filename = 'result\\' + model_name + '_DetectionResult.txt'
    with open(result_filename, 'w') as file:
        pass

    # 设定时间及空间条件、设定数据集
    site_list = ['Xintian20221017', 'Yuxi20230411', 'Liangshan20200507', 'Xichang20200330', 'Chongqing20220821', 'Dali20240315']
    SingleFireEventDetection()  # 单一火灾事件检测

