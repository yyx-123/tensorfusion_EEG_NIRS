import pickle

from scipy.io import loadmat
from scipy import signal
import numpy as np
from tqdm import tqdm

winSize = 3         # 3秒的时间窗
stride = 1          # 滑动窗口的stride为1秒

# 设置好一个[10Hz, 80Hz]的带通滤波器
lb = 10
hb = 80
fs = 200
b, a = signal.butter(5, [2 * lb / fs, 2 * hb / fs], 'bandpass')   #配置滤波器 5 表示滤波器的阶数

for subId in tqdm(range(6, 30)):
    if subId < 10:
        subId = "0" + str(subId)
    else:
        subId = str(subId)
    cnt = loadmat('D:\\fnirs dataset\eeg-fnirs dataset_mi_ma\eeg\subject ' + subId + '\with occular artifact\cnt.mat')
    mrk = loadmat('D:\\fnirs dataset\eeg-fnirs dataset_mi_ma\eeg\subject ' + subId + '\with occular artifact\mrk.mat')

    mi = []
    mi_label = []
    # 每一段任务研究时间为[-10, 25],其中stride=1秒地划分长度为3秒的时间窗
    for seg in np.arange(-10, 23, stride):
        taskData = []
        labelData = []
        # trail = 0,2,4代表MI任务，trail=1,3,5代表MA任务吗，这里先做MI任务
        for trail in range(0, 6, 2):
            # 提取EEG信号并做带通滤波
            x = cnt['cnt'][0][trail]['x'][0][0]
            for ch in range(30):
                x[:, ch] = signal.filtfilt(b, a, x[:, ch])
            # 提取任务时间信息并转换成trigger
            time = mrk['mrk'][0][trail]['time'][0][0][0]
            trigger = time / 1000 * fs             # time为每一个task的起始时间，以ms为单位，转换成x中的起始位置，需要除以1000（ms）再乘以200（采样率）
            # 提取任务label
            y = mrk['mrk'][0][trail]['y'][0][0]
            y = np.transpose(y, (1, 0))
            for task in range(20):
                labelData.append(y[task])

            # 针对每个trail的20个任务
            for task in range(20):
                start = int(trigger[task])+ seg * fs
                chData = []
                for ch in range(30):
                    chData.append(x[start : start + winSize * fs, ch])
                taskData.append(chData)
        mi.append(taskData)
        mi_label.append(labelData)

    ma = []
    ma_label = []
    # 每一段任务研究时间为[-10, 25],其中stride=1秒地划分长度为3秒的时间窗
    for seg in np.arange(-10, 23, stride):
        taskData = []
        labelData = []
        # trail = 0,2,4代表MI任务，trail=1,3,5代表MA任务吗，这里先做MI任务
        for trail in range(1, 6, 2):
            # 提取EEG信号并做带通滤波
            x = cnt['cnt'][0][trail]['x'][0][0]
            for ch in range(30):
                x[:, ch] = signal.filtfilt(b, a, x[:, ch])
            # 提取任务时间信息并转换成trigger
            time = mrk['mrk'][0][trail]['time'][0][0][0]
            trigger = time / 1000 * fs  # time为每一个task的起始时间，以ms为单位，转换成x中的起始位置，需要除以1000（ms）再乘以200（采样率）
            # 提取任务label
            y = mrk['mrk'][0][trail]['y'][0][0]
            y = np.transpose(y, (1, 0))
            for task in range(20):
                labelData.append(y[task])

            # 针对每个trail的20个任务
            for task in range(20):
                start = int(trigger[task]) + seg * fs
                chData = []
                for ch in range(30):
                    chData.append(x[start: start + winSize * fs, ch])
                taskData.append(chData)
        ma.append(taskData)
        ma_label.append(labelData)

    # 保存数据
    mi = np.array(mi)               # 33 * 60 * 30 * 600
    ma = np.array(ma)
    mi_label = np.array(mi_label)
    ma_label = np.array(ma_label)
    subData = {
        'mi': mi,
        'ma': ma,
        'mi_label': mi_label,
        'ma_label': ma_label
    }
    tgtDir = 'D:\\fnirs dataset\eeg-fnirs dataset_mi_ma\eeg\\'
    with open(tgtDir + subId + '.pickle', 'wb') as file:
        pickle.dump(subData, file)


