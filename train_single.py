import pickle
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from tqdm import tqdm
from util import *
from models.model_EEG import EEG_ResCNN
from models.model_NIRS import NIRS_ResCNN, NIRS_TCN



needSave = False
if needSave:
    import xlwt

needPrint = True

needWandB = False
if needWandB:
    import wandb
    wandb.init(project="eeg_nirs")

if __name__ == '__main__':
    K = 5
    batch_size = 16
    epoch_num = 100
    # Tasks = ['ma', 'mi']
    Tasks = ['ma']
    # Types = ['eeg', 'oxy', 'deoxy']
    Types = ['oxy']
    check_every = 2

    for Task in Tasks:
        for Type in Types:
            if needSave:
                now = datetime.now().strftime("%m%d_%H%M")
                # 创建一个新的excel保存结果，四项指标各创建一个sheet保存
                workbook = xlwt.Workbook(encoding='utf-8')
                sheet_acc = workbook.add_sheet("accuracy")
                sheet_prec = workbook.add_sheet("precision")
                sheet_recl = workbook.add_sheet("recall")
                sheet_f1 = workbook.add_sheet("f1")
                # 给4个sheet标好表头
                for row in range(1, 30):
                    sheet_acc.write(row, 0, row)
                    sheet_prec.write(row, 0, row)
                    sheet_recl.write(row, 0, row)
                    sheet_f1.write(row, 0, row)
                for col in range(33):
                    sheet_acc.write(0, col + 1, str(col - 10) + 's')
                    sheet_prec.write(0, col + 1, str(col - 10) + 's')
                    sheet_recl.write(0, col + 1, str(col - 10) + 's')
                    sheet_f1.write(0, col + 1, str(col - 10) + 's')

            # 对每个被试、每段时间窗展开相同实验
            for subID in range(1, 30, 2):
                print("Task:{}, Type:{}, SubId:{}".format(Task, Type, subID))

                for seg in tqdm(range(0, 33, 2)):
                    if needPrint:
                        print('subID:{}, Task:{}, model:{}, Time window:[{}s, {}s]'.format(subID, Task, Type, seg - 10, seg - 7))

                    # 读取数据
                    if subID < 10:
                        subId = "0" + str(subID)
                    else:
                        subId = str(subID)
                    if Type == 'eeg':
                        with open('D:\\fnirs dataset\\eeg-fnirs dataset_mi_ma\\eeg\\' + subId + ".pickle", 'rb') as file:
                            data = pickle.load(file)
                            X = data[Task][seg]
                    else:
                        with open('D:\\fnirs dataset\\eeg-fnirs dataset_mi_ma\\nirs\\' + subId + ".pickle", 'rb') as file:
                            data = pickle.load(file)
                            X = data[Task + "_" + Type][seg]
                    Y = data[Task + "_label"][seg]

                    # K折交叉实验，统计4项指标，每一项指标保存每一折的最优结果
                    kf = KFold(n_splits=K)
                    kFold_acc = np.zeros(K)
                    kFold_prec = np.zeros(K)
                    kFold_recl = np.zeros(K)
                    kFold_f1 = np.zeros(K)
                    fold = -1
                    for train_index, test_index in kf.split(X):
                        # 第fold折实验
                        fold += 1

                        # 划分数据
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = Y[train_index], Y[test_index]
                        train_set = DeformedData(X_train, y_train)
                        train_dataloader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size,
                                                      shuffle=True, pin_memory=True)

                        # 加载模型，配置训练参数
                        if Type == 'eeg':
                            model = EEG_ResCNN(channel=30).cuda()
                        else:
                            model = NIRS_TCN(channel=36).cuda()
                        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
                        Loss = nn.BCELoss().cuda()

                        # 训练过程
                        for epoch in range(epoch_num):
                            # 训练
                            model.train()
                            for ite, (input, target) in enumerate(train_dataloader):
                                input = Variable(input, requires_grad=False).cuda().float()
                                target = Variable(target, requires_grad=False).cuda().float()
                                optimizer.zero_grad()
                                output, _ = model(input)
                                loss = Loss(output, target)
                                loss.backward()
                                optimizer.step()

                            # 验证
                            if epoch % check_every == 0:
                                with torch.no_grad():
                                    model.eval()
                                    # 训练集
                                    input = Variable(torch.Tensor(X_train), requires_grad=False).cuda().float()
                                    target = Variable(torch.Tensor(y_train), requires_grad=False).cuda().float()
                                    output, _ = model(input)
                                    train_loss = Loss(output, target)

                                    _, preds = output.max(1)
                                    _, target = target.max(1)
                                    preds = np.array(preds.cpu())
                                    target = np.array(target.cpu())
                                    train_acc = accuracy_score(target, preds)

                                    # 测试集
                                    input = Variable(torch.Tensor(X_test), requires_grad=False).cuda().float()
                                    target = Variable(torch.Tensor(y_test), requires_grad=False).cuda().float()
                                    output, _ = model(input)
                                    test_loss = Loss(output, target)

                                    _, preds = output.max(1)
                                    _, target = target.max(1)
                                    preds = np.array(preds.cpu())
                                    target = np.array(target.cpu())
                                    test_acc = accuracy_score(target, preds)
                                    prec = precision_score(target, preds)
                                    recl = recall_score(target, preds)
                                    f1 = f1_score(target, preds)

                                    # 保存第fold折的最好结果
                                    kFold_acc[fold] = max(kFold_acc[fold], test_acc)
                                    kFold_prec[fold] = max(kFold_prec[fold], prec)
                                    kFold_recl[fold] = max(kFold_recl[fold], recl)
                                    kFold_f1[fold] = max(kFold_f1[fold], f1)

                                    # 记录、输出信息
                                    if needPrint:
                                        print('epoch:{}, train loss:{:.4f}, train acc:{:.3f}, test loss:{:.4f}, test acc:{:.3f}'.format(
                                                epoch, train_loss, train_acc, test_loss, test_acc))
                                    if needWandB:
                                        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss})
                                        wandb.log({'train_acc': train_acc, 'test_acc': test_acc})
                                        wandb.log({'precision': prec, 'recall': recl, 'f1': f1})

                    # 记录K折实验的最终结果
                    if needSave:
                        sheet_acc.write(subID, seg + 1, kFold_acc.mean())
                        sheet_prec.write(subID, seg + 1, kFold_prec.mean())
                        sheet_recl.write(subID, seg + 1, kFold_recl.mean())
                        sheet_f1.write(subID, seg + 1, kFold_f1.mean())
                        workbook.save('results/' + Task + "_" + Type + "_" + now + '.xls')  # 形如results/ma_eeg_1012_1100.xls
