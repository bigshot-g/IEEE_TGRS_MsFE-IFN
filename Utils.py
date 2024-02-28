import numpy as np
from operator import truediv
import random
import os
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def generate_mask(sample_num, cls_num, data_label):
    SEED = random.randint(0, 2000)
    set_seed(SEED)

    N = sample_num
    cls = cls_num
    data_gt = data_label
    cou = np.zeros(shape=(20,))
    count = 0
    gt_train = np.zeros((data_gt.shape[0], data_gt.shape[1]))
    while count < (N*cls):
        x = random.randint(0, data_gt.shape[0] - 1)
        y = random.randint(0, data_gt.shape[1] - 1)
        if data_gt[(x, y)] == 1 and gt_train[(x, y)] == 0 and cou[1] < N:
            gt_train[(x, y)] = 1
            cou[1] = cou[1] + 1
            count = count + 1

        if data_gt[(x, y)] == 2 and gt_train[(x, y)] == 0 and cou[2] < N:
            gt_train[(x, y)] = 1
            cou[2] = cou[2] + 1
            count = count + 1

        if data_gt[(x, y)] == 3 and gt_train[(x, y)] == 0 and cou[3] < N:
            gt_train[(x, y)] = 1
            cou[3] = cou[3] + 1
            count = count + 1

        if data_gt[(x, y)] == 4 and gt_train[(x, y)] == 0 and cou[4] < N:
            gt_train[(x, y)] = 1
            cou[4] = cou[4] + 1
            count = count + 1

        if data_gt[(x, y)] == 5 and gt_train[(x, y)] == 0 and cou[5] < N:
            gt_train[(x, y)] = 1
            cou[5] = cou[5] + 1
            count = count + 1

        if data_gt[(x, y)] == 6 and gt_train[(x, y)] == 0 and cou[6] < N:
            gt_train[(x, y)] = 1
            cou[6] = cou[6] + 1
            count = count + 1

        if data_gt[(x, y)] == 7 and gt_train[(x, y)] == 0 and cou[7] < N:
            gt_train[(x, y)] = 1
            cou[7] = cou[7] + 1
            count = count + 1

        if data_gt[(x, y)] == 8 and gt_train[(x, y)] == 0 and cou[8] < N:
            gt_train[(x, y)] = 1
            cou[8] = cou[8] + 1
            count = count + 1

        if data_gt[(x, y)] == 9 and gt_train[(x, y)] == 0 and cou[9] < N:
            gt_train[(x, y)] = 1
            cou[9] = cou[9] + 1
            count = count + 1

        if data_gt[(x, y)] == 10 and gt_train[(x, y)] == 0 and cou[10] < N:
            gt_train[(x, y)] = 1
            cou[10] = cou[10] + 1
            count = count + 1
        if data_gt[(x, y)] == 11 and gt_train[(x, y)] == 0 and cou[11] < N:
            gt_train[(x, y)] = 1
            cou[11] = cou[11] + 1
            count = count + 1

        if data_gt[(x, y)] == 12 and gt_train[(x, y)] == 0 and cou[12] < N:
            gt_train[(x, y)] = 1
            cou[12] = cou[12] + 1
            count = count + 1

        if data_gt[(x, y)] == 13 and gt_train[(x, y)] == 0 and cou[13] < N:
            gt_train[(x, y)] = 1
            cou[13] = cou[13] + 1
            count = count + 1

        if data_gt[(x, y)] == 14 and gt_train[(x, y)] == 0 and cou[14] < N:
            gt_train[(x, y)] = 1
            cou[14] = cou[14] + 1
            count = count + 1

        if data_gt[(x, y)] == 15 and gt_train[(x, y)] == 0 and cou[15] < N:
            gt_train[(x, y)] = 1
            cou[15] = cou[15] + 1
            count = count + 1

        if data_gt[(x, y)] == 16 and gt_train[(x, y)] == 0 and cou[16] < N:
            gt_train[(x, y)] = 1
            cou[16] = cou[16] + 1
            count = count + 1

        if data_gt[(x, y)] == 17 and gt_train[(x, y)] == 0 and cou[17] < N:
            gt_train[(x, y)] = 1
            cou[17] = cou[17] + 1
            count = count + 1

        if data_gt[(x, y)] == 18 and gt_train[(x, y)] == 0 and cou[18] < N:
            gt_train[(x, y)] = 1
            cou[18] = cou[18] + 1
            count = count + 1

        if data_gt[(x, y)] == 19 and gt_train[(x, y)] == 0 and cou[19] < N:
            gt_train[(x, y)] = 1
            cou[19] = cou[19] + 1
            count = count + 1
    gt_train = data_gt * gt_train
    gt_train = gt_train.astype(int)
    return gt_train, SEED
