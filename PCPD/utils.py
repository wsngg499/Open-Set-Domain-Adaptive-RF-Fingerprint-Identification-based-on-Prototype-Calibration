import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import torch
from numpy.random import standard_normal, uniform
from sklearn.manifold import TSNE
# from wlan_utils import *
# from preamble_utils import *
# from preamble_function import *


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def localSequenceGenerate():
    # STS频域表示，频点为 -32~31，此处将52个频点外的零补全。
    S = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 0, 0, 0, 0])
    # LTS频域表示，频点为 - 32~31，此处将52个频点外的零补全。
    L = np.array([0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # 保证OFDM符号的功率值稳定
    # my_timer_figure(np.abs(L))
    S = np.sqrt(13 / 6) * S
    S_shifted = np.fft.fftshift(S)
    # my_timer_figure(S_shifted)
    # my_timer_figure(np.abs(S_shifted))
    # 通过IFFT函数将STS频域的频点顺序调整为正频率（0~31）、负频率（-32~-1）
    short_16 = np.fft.ifft(S_shifted)[:16]
    # my_timer_figure(short_16)
    short = np.tile(short_16, 10)
    short[0] = 0.5*short[0]
    short[-1] = 0.5 * short[-1]
    # my_timer_figure(short)
    L_shifted = np.fft.fftshift(L)
    # my_timer_figure(np.abs(L_shifted))
    # my_timer_figure1(np.abs(L))
    long_cp = np.fft.ifft(L_shifted)
    long1 = long_cp[32:]
    long2 = long_cp
    long = np.concatenate((long1, long2, long2))
    long[0] = 0.5*short_16[0]+0.5*long[0]
    # my_timer_figure1(long)
    short_16 = short_16 / ((np.sum(np.abs(short_16)))/16)
    # my_timer_figure(short_16)
    short = short / ((np.sum(np.abs(short)))/160)
    long = long / ((np.sum(np.abs(long)))/160)
    # my_timer_figure(short)
    # my_timer_figure(long)
    return short_16, short, long


class pkt_args:
    def __init__(self):
        self.pkt_offset = []
        self.addr2 = []
        self.cfo = []
        self.mac_type = []
        self.pkt_length = []


class PktData:
    def __init__(self):
        self.data = None
        self.dev_label = None
        self.cfo = None
        self.pktType = None
        self.pktLen = None
        self.addr2 = None
        self.date = None
        self.temperature = None
        self.app_label = None
        self.distance = None
        self.position = None


def wifiDetection(rx, packet, process_block_length, device_mac, date, temperature, app_label, distance, position, threshold=120, slicelength_left=200, slicelength_right=1200):
    pkt_list = []
    xn, short, long = localSequenceGenerate()
    stf_winLen = 16
    # xn = short / (np.sum(np.abs(short)) / 16)
    for k in range(len(packet.pkt_offset)):
        pkt_data = PktData()
        if int(packet.pkt_offset[k]) <= slicelength_left or int(packet.pkt_offset[k]) >= process_block_length - slicelength_right:
            continue
        pkt_data_k = rx[int(packet.pkt_offset[k]) - slicelength_left:int(packet.pkt_offset[k]) + int(packet.pkt_length[k])+50+32]

        r = []
        # my_timer_figure(pkt_data_k)
        for n in range(len(pkt_data_k) - stf_winLen):
            win_rx_data = pkt_data_k[n: n + stf_winLen]
            yn = win_rx_data / (np.sum(np.abs(win_rx_data)) / 16)
            r.append(np.sum(yn * np.conj(xn)))  # 相关性系数
        r_sum = []
        for m in range(170, 230):
            # r_sum = np.sum(np.abs(r[m]), np.abs(r[m+8]), np.abs(r[m+8*2], np.abs(r[m+8*3], np.abs(r[m+8*4], np.abs(r[m+8*5], np.abs(r[m+8*6], np.abs(r[m+8*7], np.abs(r[m+8*8], np.abs(r[m+8*9]))
            if m + 9 * 16 < len(r):  # 确保后面还有 9 个元素
                elements = [np.abs(r[m + j * 16]) for j in range(10)]
                r_sum.append(np.sum(elements))
            r_max = np.max(r_sum)
            # r_max_idx = np.argmax(r_sum) + 120
        # my_timer_figure(pkt_data_k[200:360])
        # print("123")
        if r_max > threshold:
            # my_timer_figure(pkt_data_k[200:360])
            # 粗时间同步
            # r_short = []
            # rx_lstf_coarse = pkt_data_k[slicelength_left - 64:slicelength_left + 160 + 64]  # 截取出288个采样点用于同步计算
            # Rxx_short = np.sum(np.abs(short) ** 2)
            # for m in range(128):
            #     yn = rx_lstf_coarse[m:m + 160]
            #     Ryy_short = np.sum(np.abs(yn) ** 2)
            #     r_short.append(np.sum((yn * np.conj(short)) / (np.sqrt(Rxx_short * Ryy_short))))
            # coarse_offset = np.argmax(np.abs(r_short))
            # # my_timer_figure(np.abs(r_short))
            # rx_data = pkt_data_k[slicelength_left-64+coarse_offset-132:slicelength_left-64+coarse_offset+int(packet.pkt_length[k])+50+32]
            # my_timer_figure(rx_data[132-5:132+320])  # 粗同步后的lstf
            #
            # # 粗频偏补偿
            # fs = 20e6
            # delta_i = []
            # for n in range(9):
            #     sita = []
            #     for L in range(16):
            #         idx = 132+16*n+L
            #         sita.append(np.angle(rx_data[idx+16] * np.conj(rx_data[idx])))
            #     delta_i.append(np.sum(sita) / 16)
            # delta = np.sum(delta_i) / 9
            # delta_f = delta * fs / (2 * np.pi * 16)
            # rx_data = freqCompensation(rx_data, delta_f)
            # # my_timer_figure(rx_data[132-10:132+320])
            # # my_timer_figure(rx_data[260:420])
            #
            # # 细时间同步
            # r_long = []
            # rx_lltf_coarse = rx_data[132 - 32 + 160:132 + 33 + 320]
            # # my_timer_figure1(rx_lltf_coarse)
            # Rxx_long = np.sum(np.abs(long) ** 2)
            # for m in range(65):
            #     yn = rx_lltf_coarse[m:m + 160]
            #     Ryy_long = np.sum(np.abs(yn) ** 2)
            #     r_long.append(np.sum((yn * np.conj(long)) / (np.sqrt(Rxx_long * Ryy_long))))
            # fine_offset = np.argmax(np.abs(r_long))
            # rx_data = rx_data[132 - 32 + fine_offset - 100:132 - 32 + fine_offset + int(packet.pkt_length[k])+50]  # 设置截取数据的长度
            # # my_timer_figure(rx_data[100:260])
            # # my_timer_figure(rx_data[260:420])
            # # 细频偏补偿
            # sita_long = []
            # for L in range(64):
            #     sita_long.append(np.angle(rx_data[100 + 160 + 32 + 64 + L] * np.conj(rx_data[100 + 160 + 32 + L])))
            # delta_long = np.sum(sita_long) / 64
            # delta_f_long = delta_long * fs / (2 * np.pi * 64)
            # rx_data = freqCompensation(rx_data, delta_f_long)
            # my_timer_figure(rx_data[100 - 5:100 + 320])
            #
            # # 相偏补偿
            # preamble = np.concatenate((short, long))
            # angle_i = np.angle(rx_data[100:100+320] * np.conj(preamble))
            # rx_data[100:100+320] = rx_data[100:100+320] * np.exp(1j * (-angle_i))
            # my_timer_figure(rx_data[100-5:100+320])


            # my_timer_figure(rx_data[100-10:420])
            # my_timer_figure(rx_data)

            rx_data = rx[int(packet.pkt_offset[k])-100:int(packet.pkt_offset[k]) + int(packet.pkt_length[k])+50]
            # my_timer_figure(rx_data)
            my_timer_figure(freqCompensation(rx_data, float(packet.cfo[k])))
            pkt_data.data = rx_data
            pkt_data.pktLen = packet.pkt_length[k]
            pkt_data.pktType = packet.mac_type[k]
            pkt_data.addr2 = packet.addr2[k]
            pkt_data.cfo = packet.cfo[k]
            pkt_data.date = date
            pkt_data.temperature = temperature
            pkt_data.app_label = app_label
            pkt_data.distance = distance
            pkt_data.position = position
            pkt_data.dev_label = np.array(np.where([list(d.values())[0] == packet.addr2[k] for d in device_mac])[0], dtype=int)
            pkt_list.append(pkt_data)
            # pkt_data = np.concatenate((pkt_data, rx_data.reshape(1, -1)), axis=0)
            # print("123")
    return pkt_list


def freq_coarse2():
    # 粗频偏补偿
    fs = 20e6
    delta_i = []
    rx_new = np.empty(0)
    for n in range(10):
        sita = []
        for L in range(16):
            idx = 132 + 16 * n + L
            sita.append(np.angle(rx_data[idx] * np.conj(xn[L])))
            delta_f = sita[L] * fs / (2 * np.pi * 16)
            rx_data_i = rx_data[idx] * np.exp(1j * 2 * np.pi * (-delta_f) / fs)
            rx_new = np.append(rx_new, rx_data_i)
            # print("123")
        # delta_i.append(np.sum(sita) / 16)
        # delta_f = delta_i[n] * fs / (2 * np.pi * 16)
        # rx_data_i = freqCompensation(rx_data[132+16*n:132+16*n+16], delta_f)
        # rx_new = np.concatenate((rx_new, rx_data_i), axis=0)
    my_timer_figure(rx_new)
    delta = np.sum(delta_i) / 9
    delta_f = delta * fs / (2 * np.pi * 16)

    rx_data = freqCompensation(rx_data, delta_f)
    # my_timer_figure(rx_data[132-10:132+320])
    # my_timer_figure(rx_data[260:420])

def my_timer_figure(data):
    plt.figure()
    plt.plot(data)
    plt.title('时域波形图')
    plt.xlabel('时间')
    plt.ylabel('幅度')
    plt.show()

# 粗频偏估计
def coarseCfoEstimate(lstf, fs=20e6):
    corrOffset = 0.75       # 默认相关性偏移参数
    fft_len = 64
    nums_lstf = 160
    # 粗略的CFO估计，假设每个FFT周期有4次重复。
    # M为每个重复的样本数，GI为保护间隔长度，S为L-STF的最大有效部分长度，
    # N为输入信号的样本数。
    M = int(fft_len / 4)         # 周期:16
    GI = int(fft_len / 4)        # 保护间隔长度:16
    S = int(M * 9)               # L-STF的最大有效部分长度: 144

    offset = round(corrOffset * GI)
    lstf_used = lstf[offset:offset+min(S, nums_lstf-offset)]

    # 频偏估计
    cx = lstf_used[:-M]
    sx = lstf_used[M:]
    res = np.conj(cx) @ sx
    coarse_foffset = np.angle(res) / (2 * np.pi) * fs / M
    return coarse_foffset


# 频偏补偿
def freqCompensation(preamble, coarseFreqOffset, sr=20e6):
    fs = sr
    t = np.arange(len(preamble)) / fs
    y = preamble * np.exp(1j * 2 * np.pi * (-coarseFreqOffset) * t)
    # my_timer_figure(y)
    return y


def freqCfoEstimate(preamble, fs=20e6):
    coarse_foffset = coarseCfoEstimate(preamble[:160])
    preamble = freqCompensation(preamble, coarse_foffset)
    fine_foffset = fineCfoEstimate(preamble[160:])
    return coarse_foffset + fine_foffset

# 细频偏估计
def fineCfoEstimate(lltf, fs=20e6):
    corrOffset = 0.75       # 默认相关性偏移参数
    fft_len = 64
    nums_lstf = 160
    # 粗略的CFO估计，假设每个FFT周期有4次重复。
    # M为每个重复的样本数，GI为保护间隔长度，S为L-STF的最大有效部分长度，
    # N为输入信号的样本数。
    M = int(fft_len)         # 重复的样本数:64
    GI = int(fft_len / 2)        # 保护间隔长度:32
    S = int(M * 2)               # L-ltf的最大有效部分长度: 128

    offset = round(corrOffset * GI)
    lltf_used = lltf[offset:offset+min(S, nums_lstf-offset)]

    # 频偏估计
    cx = lltf_used[:-M]
    sx = lltf_used[M:]
    res = np.conj(cx) @ sx
    fine_foffset = np.angle(res) / (2 * np.pi) * fs / M
    return fine_foffset


def normalized(data):
    return (data-data.min()) / (data.max()-data.min())


def normalized_2d(data):
    norm_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        norm_data[i] = (data[i]-data[i].min()) / (data[i].max()-data[i].min())
    return norm_data


def data_show():
    m = 7
    data_root = "/home/data1/WZM/Dataset/temperature//"
    data_path = [
        data_root + "2024_04_25-26/train_data.pkt",
        data_root + "2024_04_28-30/train_data.pkt",
        data_root + "2024_05_06-07/train_data.pkt",
        data_root + "2024_05_08-09/train_data.pkt",
        data_root + "2024_05_17-20/train_data.pkt"
    ]
    idx1 = [8, 12, 16, 20, 24, 28, 36, 40, 44, 48, 52, 56]
    onedev_alldays = []
    for n in range(5):
        with open(data_path[n], "rb") as f:
            pkt_data = pickle.load(f)
            data_np = np.zeros((len(pkt_data)//10, 320), dtype=complex)
            cfo = np.zeros(len(pkt_data)//10, dtype=float)
            label = np.zeros(len(pkt_data)//10, dtype=int)
            count = 0
            for i, data in enumerate(pkt_data):
                if data.dev_label == m:
                    data_i = data.data[100:100 + 320]
                    cfo[count] = float(data.cfo)
                    # my_timer_figure(data_i)
                    data_i = freqCompensation(data_i, cfo[count])
                    # my_timer_figure(data_i)
                    data_np[count] = data_i
                    label[count] = data.dev_label
                    count += 1
        onedev_alldays.append(data_np)
    plt.figure(figsize=(16, 8))
    color = [
        "red", "orange", "yellow", "green", "blue"
    ]
    for n, data_every in enumerate(onedev_alldays):
        sts1 = data_every[:, 16:16 * 5]
        sts2 = data_every[:, 16 * 5:16 * 9]
        lts1 = data_every[:, 192:192 + 64]
        lts2 = data_every[:, 192 + 64:192 + 64 * 2]
        a = normalized_2d(np.abs(np.fft.fftshift(np.fft.fft(sts1))))
        b = normalized_2d(np.abs(np.fft.fftshift(np.fft.fft(sts2))))
        c = normalized_2d(np.abs(np.fft.fftshift(np.fft.fft(lts1))))
        d = normalized_2d(np.abs(np.fft.fftshift(np.fft.fft(lts2))))
        sts = (a + b) / 2
        lts = (c + d) / 2
        rff1 = np.log(sts) - np.log(lts)
        rff = rff1[:, idx1]
        days = "day-" + str(n)
        # plt.plot(rff1[0], color='green', marker='d', label='STS1')  # 绿色方块线
        # plt.plot(rff[0], color='black', marker='*', label='RFF')  # 黑色星号

        # 使用 t-SNE 进行降维至二维
        tsne = TSNE(n_components=2)
        X_2d = tsne.fit_transform(rff)

        # 定义每个数据点对应的日期
        # dates = [0] * len(data_23_1207_train) + [1] * len(data_23_1208_train)

        # 定义颜色映射
        # color_map = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'green', 4: 'cyan', 5: 'blue'}
        # plt.figure(figsize=(12, 8))
        # 绘制散点图
        for i in range(len(X_2d)):
            plt.scatter(X_2d[i, 0], X_2d[i, 1], c=color[n])

        plt.title(f"Device: {m}")
        # 添加图例
        # for day, color in color:
        plt.scatter([], [], c=color[n], label=days)
        plt.legend()

    plt.show()
    print("123")


def data_show2():
    print("123")


if __name__ == '__main__':

    data_show()

    # with open("D:\Prince\Project_code\data\Datasets//temperature//24_04_22//test_data.pkt", 'rb') as f1:
    #     data = pickle.load(f1)
    print("123")
    # data = np.load("D://Prince//Project_code//data//devices_nums20_length400//24_03_28//train_data.npy")
    # data1 = data[0][80:-1]
    # # my_timer_figure(data[0][:-1])
    # my_timer_figure(data1)
    # coarse_foffset = coarseCfoEstimate(data1[:160])
    # y1 = freqCompensation(data1, coarse_foffset)
    # # lltfOffset = wlanSymbolTimingEstimate(y)
    # fine_foffset = fineCfoEstimate(data1[160:])
    # y2 = freqCompensation(y1, fine_foffset)
    # cfoOffset = coarse_foffset + fine_foffset
    # y3 = freqCompensation(data1, cfoOffset)



