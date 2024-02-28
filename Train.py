import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.utils.data as Data
from Model import model
from Loss import Loss_orthogonal, Loss_distance
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from Utils import aa_and_each_accuracy, set_seed
from thop import profile, clever_format
from Data_process import data_process


train_HSI, train_LiDAR, label_train, test_HSI, test_LiDAR, label_test, SEED = data_process()  # Replace the code that loads the data

start_time = time.time()
print("Data convertng...")

train_HSI = torch.from_numpy(np.array(train_HSI))
train_LiDAR = torch.from_numpy(np.array(train_LiDAR))
label_train = torch.from_numpy(np.array(label_train)).long()
label_train = label_train.view(-1, 1)

test_HSI = torch.from_numpy(np.array(test_HSI))
test_LiDAR = torch.from_numpy(np.array(test_LiDAR))
label_test = torch.from_numpy(np.array(label_test)).long()
label_test = label_test.view(-1, 1)

print("End data convert:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "耗时%.3fs" % (time.time() - start_time))

set_seed(SEED)
cls = 12
patch_size = 11
hsi_channel = 126
lidar_channel = 1
batch_size_TR = 128
batch_size_TE = 2000

train_data = Data.TensorDataset(train_HSI, torch.unsqueeze(train_LiDAR, dim=3), label_train)
test_data = Data.TensorDataset(test_HSI, torch.unsqueeze(test_LiDAR, dim=3), label_test)
train_iter = Data.DataLoader(
        dataset=train_data,
        batch_size=batch_size_TR,
        shuffle=True,
        num_workers=0,
    )
test_iter = Data.DataLoader(
        dataset=test_data,
        batch_size=batch_size_TE,
        shuffle=False,
        num_workers=0,
    )

net = model(hsi_channel, lidar_channel, cls)
KL = nn.KLDivLoss(reduction='batchmean')
criterion_cls = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999)) #0.0001-0.000001
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15, eta_min=1e-6)

print("Start Traning...")
start_time_Tr = time.time()
net.cuda()
for epoch in range(50):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    net.train()
    for i, (H, L, y) in enumerate(train_iter):
        H = H.cuda()
        L = L.cuda()
        y = y.cuda()
        opt.zero_grad()
        x_H, x_a, x_b, x_L, result = net(H, L)

        # loss calculation
        loss_ort = Loss_orthogonal()(x_a, x_b)

        x_L1 = F.log_softmax(x_L, dim=1)
        x_H1 = F.softmax(x_H, dim=1)
        KL_LH = KL(x_L1, x_H1)
        x_H2 = F.log_softmax(x_H, dim=1)
        x_L2 = F.softmax(x_L, dim=1)
        KL_HL = KL(x_H2, x_L2)
        loss_KL = (KL_LH + KL_HL)*0.5

        loss_cls = criterion_cls(result, y.squeeze(dim=1))

        M = (F.softmax(x_L, dim=1) + F.softmax(x_H, dim=1)) * 0.5
        P_L = F.log_softmax(x_L, dim=1)
        P_H = F.log_softmax(x_H, dim=1)
        loss_JS = (KL(P_L, M) + KL(P_H, M)) * 0.5

        loss_dis = Loss_distance()(x_H, x_L, 'cosine')  # euclidean, cosine

        loss = loss_cls + loss_ort * 0.3 + loss_KL * 0.001
        loss.backward()
        opt.step()
        #scheduler_G.step()
        print('[%d, %5d] loss:%.3f, loss_cls:%.3f, loss_ort:%.3f, loss_KL:%.3f'\
                % (epoch + 1, (i + 1) * batch_size_TR, loss, loss_cls, loss_ort, loss_KL))
    print("第%d个epoch的学习率：G:%f" % (epoch+1, opt.param_groups[0]['lr']))
print("Train End:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "耗时%.3fs" % (time.time() - start_time_Tr))
print('ALL Parameters:', sum(p.numel() for p in net.parameters()))
torch.save(net.state_dict(), './test.pth')

#net.load_state_dict(torch.load('test.pth'))
print("Start testing...")
#net.cuda()
net.eval()
start_time_Te = time.time()
correct = 0
total = len(label_test)
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
pre = np.array([], dtype=np.int32)
HSI = np.array([], dtype=np.int32)
LiDAR = np.array([], dtype=np.int32)
for i, (H, L, y) in enumerate(test_iter):
    H = H.cuda()
    L = L.cuda()
    y = y.cuda()
    with torch.no_grad():
        x_H, x_a, x_b, x_L, result = net(H, L)

    _, predicted = torch.max(result.data, 1)
    cou = 0
    cou += (predicted == y.squeeze(dim=1)).sum()
    correct += cou
    predicted = np.array(predicted.cpu())
    pre = np.concatenate([pre, predicted], 0)
    print("[%d, %5d] Accuracy:%.4f %%" % ((i+1)*batch_size_TE, total, 100 * (correct.float() / ((i + 1)*batch_size_TE))))
print("Overall Accuracy：%.4f %%" % (100*correct.float()/total))
print("Test End, Cost time:", time.time()-start_time_Te, "s")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
print("SEED:", SEED)


label = label_test
pre = np.asarray(pre, dtype=np.int64)
confusion_matrix = confusion_matrix(label, pre)
print(confusion_matrix)
oa = accuracy_score(label, pre)
print("OA:", oa)
each_acc, aa = aa_and_each_accuracy(confusion_matrix)
print(each_acc)
print("AA:", aa)
kappa = cohen_kappa_score(label, pre)
print("Kappa:", kappa)
np.savetxt("confusion_matrix", confusion_matrix, fmt="%5d")
scio.savemat('label_pre.mat', {'label': pre})
input1 = torch.randn(1, patch_size, patch_size, hsi_channel).cuda()
input2 = torch.randn(1, patch_size, patch_size, lidar_channel).cuda()
flops, params = profile(net, inputs=(input1, input2))
flops, params = clever_format([flops, params], "%.3f")
print("FLOPs:", flops, "Parameters:", params)
print("Time:", time.time() - start_time, "s")

