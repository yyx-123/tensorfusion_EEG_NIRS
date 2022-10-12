from scipy.io import loadmat
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import time
import argparse
from myModel_hybrid import LF, TF, PF
from util import *
import pickle
import xlwt
import wandb
from datetime import datetime
from tqdm import tqdm

needPrint = False
needWandB = False
if needWandB:
    wandb.init(project="eeg_nirs")


if __name__ == '__main__':
    K = 5
    batch_size = 16
    epoch_num = 100
    check_every = 2
    Task = 'ma'
    models = ['PF', 'LF', 'TF']

    # 写excel记录结果
    workbook = xlwt.Workbook(encoding='utf-8')
    now = datetime.now().strftime("%m%d_%H%M")
    for modelName in models:
        # 新的模型，创建一个新的sheet保存结果
        print(modelName + ":")
        sheet = workbook.add_sheet(modelName)
        for row in range(1, 30):
            sheet.write(row, 0, row)
        for col in range(33):
            sheet.write(0, col + 1, str(col - 10) + 's')

        # id似乎是被试id的意思。29个被试，因此id范围为[0,29)
        for id in tqdm(range(1, 30, 2)):
            # 每个session有35秒（任务前10秒+任务中10秒+任务后15秒），截取长度为3秒，步长为1秒的时间窗：[0-3],[1-4],[2-5],...[32-35]，总共有33段时间窗。i为时间窗起始编号[0, 32]
            for i in range(0, 33, 2):
                if needPrint:
                    print('subID:{}, Task:{}, model:{}, Time window:[{}s, {}s]'.format(id, Task, modelName, i - 10, i - 7))

                # 读取数据
                if id < 10:
                    subId = "0" + str(id)
                else:
                    subId = str(id)
                with open('/home/ubuntu/data1T/YYX/dataset_mima/eeg/' + subId + ".pickle", 'rb') as file:
                    eeg_data = pickle.load(file)
                with open('/home/ubuntu/data1T/YYX/dataset_mima/nirs/' + subId + ".pickle", 'rb') as file:
                    nirs_data = pickle.load(file)

                X_EEG = eeg_data[Task][i]                       # 600*30*60->60*30*600 (batch*channel*times)
                X_oxy = nirs_data[Task + "_oxy"][i]
                X_deoxy = nirs_data[Task + "_deoxy"][i]
                Y = eeg_data[Task + "_label"][i]            # 2*60->60*2

                # K折交叉实验
                kf = KFold(n_splits=K)
                kFold_results = []
                for train_index, test_index in kf.split(X_EEG):
                    # 划分数据
                    X_train_EEG, X_test_EEG = X_EEG[train_index], X_EEG[test_index]
                    X_train_oxy, X_test_oxy = X_oxy[train_index], X_oxy[test_index]
                    X_train_deoxy, X_test_deoxy = X_deoxy[train_index], X_deoxy[test_index]
                    y_train, y_test = Y[train_index], Y[test_index]
                    train_set = DeformedTripleData(X_train_EEG, X_train_oxy, X_train_deoxy, y_train)
                    train_dataloader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size, shuffle=True, pin_memory=True)

                    # 加载模型，配置训练参数
                    if modelName == 'LF':
                        model = LF(EEG_channel=30, NIRS_channel=36, rank=16).cuda()
                    elif modelName == 'TF':
                        model = TF(EEG_channel=30, NIRS_channel=36, rank=16).cuda()
                    elif modelName == 'PF':
                        model = PF(EEG_channel=30, NIRS_channel=36, rank=16).cuda()
                    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.1)
                    Loss = nn.BCELoss().cuda()

                    # 训练过程
                    best_test_acc = 0
                    for epoch in range(epoch_num):
                        model.train()
                        for _, (input1, input2, input3, target) in enumerate(train_dataloader):
                            input1 = Variable(input1, requires_grad=False).cuda().float()
                            input2 = Variable(input2, requires_grad=False).cuda().float()
                            input3 = Variable(input3, requires_grad=False).cuda().float()
                            target = Variable(target, requires_grad=False).cuda().float()
                            optimizer.zero_grad()
                            output = model(input1, input2, input3)
                            loss = Loss(output, target)
                            loss.backward()
                            optimizer.step()

                        # 周期性地check一下
                        if epoch % check_every == 0:
                            with torch.no_grad():
                                model.eval()
                                # 训练集
                                input1 = Variable(torch.Tensor(X_train_EEG), requires_grad=False).cuda().float()
                                input2 = Variable(torch.Tensor(X_train_oxy), requires_grad=False).cuda().float()
                                input3 = Variable(torch.Tensor(X_train_deoxy), requires_grad=False).cuda().float()
                                target = Variable(torch.Tensor(y_train), requires_grad=False).cuda().float()
                                output = model(input1, input2, input3)
                                train_loss = Loss(output, target)

                                _, preds = output.max(1)
                                _, target = target.max(1)
                                train_correct = float(preds.eq(target).sum()) / X_train_EEG.shape[0]

                                # 测试集
                                input1 = Variable(torch.Tensor(X_test_EEG), requires_grad=False).cuda().float()
                                input2 = Variable(torch.Tensor(X_test_oxy), requires_grad=False).cuda().float()
                                input3 = Variable(torch.Tensor(X_test_deoxy), requires_grad=False).cuda().float()
                                target = Variable(torch.Tensor(y_test), requires_grad=False).cuda().float()
                                output = model(input1, input2, input3)
                                test_loss = Loss(output, target)

                                _, preds = output.max(1)
                                _, target = target.max(1)
                                test_correct = float(preds.eq(target).sum()) / X_test_EEG.shape[0]
                                if best_test_acc < test_correct:
                                    best_test_acc = test_correct

                                # 记录信息
                                if needPrint:
                                    print('epoch:{}, train loss:{:.4f}, train acc:{:.3f}, test loss:{:.4f}, test acc:{:.3f}'.format(epoch, train_loss, train_correct, test_loss, test_correct))
                                if needWandB:
                                    wandb.log({'epoch':epoch, 'train_loss':train_loss, 'test_loss':test_loss})
                                    wandb.log({'epoch':epoch, 'train_acc':train_correct, 'test_acc':test_correct})

                    # 记录当前这一折结果
                    kFold_results.append(best_test_acc)

                # 记录K折实验的最终结果
                kFold_results = np.array(kFold_results)
                sheet.write(id, i + 1, kFold_results.mean())
                workbook.save('results/rst' + now + '.xls')

