import torch
import numpy as np
import netCDF4 as nc
import os


def ReadData(L_s, R_s):
    # 参数设定
    bands = 16  # 波段数
    num = 1  # 时次
    L_e = L_s + 1
    R_e = R_s + 1

    # 时间列表读取
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
    time_len = len(time_list)

    # 读取数据及数据裁剪
    tiff_list = []
    for d in range(d_s, d_e + 1):
        for t in range(time_len):
            timeline = time_list[t].strip().split(':')
            h_str = timeline[0]
            m_str = timeline[1]
            if month < 10:
                month_str = "0" + str(month)
            else:
                month_str = str(month)
            if d < 10:
                d_str = "0" + str(d)
            else:
                d_str = str(d)

            # 读入数据
            filename = ""
            filename = r'H:\\' + str(year) + month_str + d_str + '\\NC_H0' + wxbb + '_' + str(year) + month_str + d_str + '_' + h_str + m_str + '_R21_FLDK.06001_06001.nc'

            tiff_list.append([])
            if os.path.exists(filename):
                nc_data = nc.Dataset(filename)

                # 设定输出数据数组
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['albedo_01'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['albedo_02'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['albedo_03'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['albedo_04'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['albedo_05'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['albedo_06'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_07'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_08'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_09'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_10'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_11'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_12'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_13'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_14'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_15'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))
                tiff_list[t + (d-d_s) * time_len].append(torch.Tensor(nc_data['tbb_16'][L_s * L_arrange + L_bia:L_e * L_arrange + L_bia, R_s * R_arrange + R_bia:R_e * R_arrange + R_bia]))

            # 读取进度可视化
            process = "已读取：" + str(num) + "/" + str(time_len * (d_e - d_s + 1)) + "，  时间：" + month_str + d_str + " " + h_str + m_str
            num = num + 1
            print(process)

    # 数据补全
    for t in range(len(tiff_list)):
        if len(tiff_list[t]) == 0:
            n = 1
            for i in range(1, 10):
                if len(tiff_list[t + i]) > 0:
                    n = i
                    break
            for i in range(16):
                tiff_list[t].append((tiff_list[t - 1][i] + tiff_list[t + n][i]) / 2.0)

    data = []
    for t in range(len(tiff_list)):
        data.append(torch.stack(tiff_list[t]))

    data = torch.stack(data)

    if month < 10:
        month_str = "0" + str(month)
    else:
        month_str = str(month)

    if d_s < 10:
        d_s_str = "0" + str(d_s)
    else:
        d_s_str = str(d_s)
    torch.save(data, site + str(year) + month_str + d_s_str + 'FireData.pt')
    print("数据读取完成！")


#site, wxbb, year, month, d_s, d_e = "Xichang", '8', 2020, 3, 30, 30  # 地址、卫星版本、年月、起止日、缺失时段数【西昌】
#L_s, R_s, L_bia, R_bia = 80, 55, 5, -6  # 20x20
site, wxbb, year, month, d_s, d_e = "Liangshan", '8', 2020, 5, 7, 7  # 地址、卫星版本、年月、起止日、缺失时段数【凉山】
L_s, R_s, L_bia, R_bia = 79, 55, 6, 7  # 20x20

L_arrange = 20  # 行范围
R_arrange = 20  # 列范围

ReadData(L_s, R_s)


