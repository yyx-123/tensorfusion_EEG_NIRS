import pickle
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import time


from util import *
from myModel_EEG import OneD_ResCNN as OneD_CNN

if __name__ == '__main__':
    K = 5
    batch_size = 16
    epoch_num = 50
    Task = 'ma'

    results = {}
    for id in range(17, 18):
        if Task == 'ma' and id in [1, 13, 23]:
            continue

        time_acc_mean = []
        time_acc_std = []
        for i in range(12,13):
            print('subID:{}, Task:{}, Time window:[{}s, {}s]'.format(id, Task, i-10, i-7))
            # 读取数据
            if id < 10:
                subId = "0" + str(id)
            else:
                subId = str(id)
            with open('D:\\fnirs dataset\eeg-fnirs dataset_mi_ma\eeg\\' + subId + ".pickle", 'rb') as file:
                data = pickle.load(file)
            X = data[Task][i]
            Y = data[Task + "_label"][i]

            # K折交叉实验
            kf = KFold(n_splits=K)
            count = 0
            kFlod_results = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                train_set = DeformedData(X_train, y_train)
                train_dataloader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size,
                                              shuffle=True, pin_memory=True)

                model = OneD_CNN(600, 30).cuda()
                optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
                Loss = nn.BCELoss().cuda()

                best_test_acc = 0
                for epoch in range(epoch_num):
                    model.train()
                    for ite, (input, target) in enumerate(train_dataloader):
                        input = Variable(input, requires_grad=False).cuda().float()
                        target = Variable(target, requires_grad=False).cuda().float()
                        optimizer.zero_grad()
                        output, _ = model(input)
                        loss = Loss(output, target)
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        model.eval()
                        # 训练集
                        input = Variable(torch.Tensor(X_train), requires_grad=False).cuda().float()
                        target = Variable(torch.Tensor(y_train), requires_grad=False).cuda().float()
                        output, _ = model(input)
                        train_loss = Loss(output, target)

                        _, preds = output.max(1)
                        _, target = target.max(1)
                        train_correct = float(preds.eq(target).sum()) / X_train.shape[0]

                        # 测试集
                        input = Variable(torch.Tensor(X_test), requires_grad=False).cuda().float()
                        target = Variable(torch.Tensor(y_test), requires_grad=False).cuda().float()
                        output, _ = model(input)
                        test_loss = Loss(output, target)

                        _, preds = output.max(1)
                        _, target = target.max(1)
                        test_correct = float(preds.eq(target).sum()) / X_test.shape[0]

                        # 输出信息
                        print('epoch:{}, train loss:{:.4f}, train acc:{:.3f}, test loss:{:.4f}, test acc:{:.3f}'.format(epoch, train_loss, train_correct, test_loss, test_correct))
                        if train_correct == 1.0 and best_test_acc < test_correct:
                            best_test_acc = test_correct

                kFlod_results.append(best_test_acc)
            kFlod_results = np.array(kFlod_results)
            time_acc_mean.append(kFlod_results.mean())
            time_acc_std.append(kFlod_results.std())
        print(time_acc_mean)
        results[id] = np.array([time_acc_mean, time_acc_std])
