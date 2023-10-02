import os
import argparse
import random
import numpy as np
from Solver import Solver
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main(config):

    folder_path = {
        'style': './dataset/style/',
        'content': './dataset/content/',
        'stylized': './dataset/stylized/',
    }

    sr_srcc_all = np.zeros(config.train_test_num, dtype=float)
    sr_krcc_all = np.zeros(config.train_test_num, dtype=float)
    sr_plcc_all = np.zeros(config.train_test_num, dtype=float)
    sr_rmse_all = np.zeros(config.train_test_num, dtype=float)


    print('Training and testing on %s dataset for %d rounds...' % (config.dataset_style, config.train_test_num))
    for p in range(config.train_test_num):
        print('Round %d' % (p+1))
        Round_num=p+1
        # Randomly select 80% groups for training and the rest for testing
        # 贴测试和训练的组标签
        group_label = []
        type_index=[]

        # type_index = [131, 79, 104, 52, 117, 116, 28, 41, 53, 137, 65, 35, 128, 148, 25, 46, 22, 10, 107, 110, 7, 31, 5, 100, 57, 135, 105, 70, 32, 127,
        #               120, 54, 18, 55, 118, 94, 92, 115, 1, 75, 80, 69, 114, 20, 68, 138, 147, 126, 84, 29, 142, 96, 112, 43, 113, 146, 15, 88, 21, 136,
        #               111, 133, 91, 134, 50, 13, 58, 144, 48, 47, 59, 9, 27, 90, 49, 30, 130, 83, 19, 143, 141, 33, 140, 129, 71, 40, 23, 89, 2, 145,
        #               97, 51, 12, 125, 3, 24, 67, 64, 45, 123, 99, 60, 139, 109, 63, 36, 124, 108, 98, 34, 56, 11, 39, 149, 78, 106, 42, 87, 37, 4,
        #               66, 26, 132, 119, 76, 8, 101, 17, 93, 6,103, 16, 44, 61, 95, 72, 86, 38, 121, 122, 82, 77, 73, 85, 14, 0, 62, 74, 102, 81]

        # type_index = [ 120, 54, 18, 55, 118, 94, 92, 115, 1, 75, 80, 69, 114, 20, 68, 138, 147, 126, 84, 29, 142, 96, 112, 43, 113, 146, 15, 88, 21, 136,
        #               111, 133, 91, 134, 50, 13, 58, 144, 48, 47, 59, 9, 27, 90, 49, 30, 130, 83, 19, 143, 141, 33, 140, 129, 71, 40, 23, 89, 2, 145,
        #               97, 51, 12, 125, 3, 24, 67, 64, 45, 123, 99, 60, 139, 109, 63, 36, 124, 108, 98, 34, 56, 11, 39, 149, 78, 106, 42, 87, 37, 4,
        #               66, 26, 132, 119, 76, 8, 101, 17, 93, 6,103, 16, 44, 61, 95, 72, 86, 38, 121, 122, 82, 77, 73, 85, 14, 0, 62, 74, 102, 81,
        #               131, 79, 104, 52, 117, 116, 28, 41, 53, 137, 65, 35, 128, 148, 25, 46, 22, 10, 107, 110, 7, 31, 5, 100, 57, 135, 105, 70, 32, 127]


        # type_index = [111, 133, 91, 134, 50, 13, 58, 144, 48, 47, 59, 9, 27, 90, 49, 30, 130, 83, 19, 143, 141, 33, 140, 129, 71, 40, 23, 89, 2, 145,
        #               97, 51, 12, 125, 3, 24, 67, 64, 45, 123, 99, 60, 139, 109, 63, 36, 124, 108, 98, 34, 56, 11, 39, 149, 78, 106, 42, 87, 37, 4,
        #               66, 26, 132, 119, 76, 8, 101, 17, 93, 6,103, 16, 44, 61, 95, 72, 86, 38, 121, 122, 82, 77, 73, 85, 14, 0, 62, 74, 102, 81,
        #               131, 79, 104, 52, 117, 116, 28, 41, 53, 137, 65, 35, 128, 148, 25, 46, 22, 10, 107, 110, 7, 31, 5, 100, 57, 135, 105, 70, 32, 127,
        #                120, 54, 18, 55, 118, 94, 92, 115, 1, 75, 80, 69, 114, 20, 68, 138, 147, 126, 84, 29, 142, 96, 112, 43, 113, 146, 15, 88, 21, 136]


        # type_index = [ 97, 51, 12, 125, 3, 24, 67, 64, 45, 123, 99, 60, 139, 109, 63, 36, 124, 108, 98, 34, 56, 11, 39, 149, 78, 106, 42, 87, 37, 4,
        #               66, 26, 132, 119, 76, 8, 101, 17, 93, 6,103, 16, 44, 61, 95, 72, 86, 38, 121, 122, 82, 77, 73, 85, 14, 0, 62, 74, 102, 81,
        #               131, 79, 104, 52, 117, 116, 28, 41, 53, 137, 65, 35, 128, 148, 25, 46, 22, 10, 107, 110, 7, 31, 5, 100, 57, 135, 105, 70, 32, 127,
        #                120, 54, 18, 55, 118, 94, 92, 115, 1, 75, 80, 69, 114, 20, 68, 138, 147, 126, 84, 29, 142, 96, 112, 43, 113, 146, 15, 88, 21, 136,
        #               111, 133, 91, 134, 50, 13, 58, 144, 48, 47, 59, 9, 27, 90, 49, 30, 130, 83, 19, 143, 141, 33, 140, 129, 71, 40, 23, 89, 2, 145]

        type_index = [66, 26, 132, 119, 76, 8, 101, 17, 93, 6,103, 16, 44, 61, 95, 72, 86, 38, 121, 122, 82, 77, 73, 85, 14, 0, 62, 74, 102, 81,
                      131, 79, 104, 52, 117, 116, 28, 41, 53, 137, 65, 35, 128, 148, 25, 46, 22, 10, 107, 110, 7, 31, 5, 100, 57, 135, 105, 70, 32, 127,
                       120, 54, 18, 55, 118, 94, 92, 115, 1, 75, 80, 69, 114, 20, 68, 138, 147, 126, 84, 29, 142, 96, 112, 43, 113, 146, 15, 88, 21, 136,
                      111, 133, 91, 134, 50, 13, 58, 144, 48, 47, 59, 9, 27, 90, 49, 30, 130, 83, 19, 143, 141, 33, 140, 129, 71, 40, 23, 89, 2, 145,
                       97, 51, 12, 125, 3, 24, 67, 64, 45, 123, 99, 60, 139, 109, 63, 36, 124, 108, 98, 34, 56, 11, 39, 149, 78, 106, 42, 87, 37, 4]

        # type_index = random.sample(range(0, 150), 150)   #随机划分训练集
        # print(type_index)
        # type_index=list(np.linspace(0,0,num = 150,dtype=int)) #固定训练集
        for j in range(150):
            e = type_index[j]
            for i in range(8):
                group_label.append(e)
        train_group_label = group_label[0:960]
        test_group_label = group_label[960:1200]

        # 训练和测试图像的序号
        # 训练序号
        func = lambda x, y, z: x * y + z
        u1 = list(np.linspace(1, 8, num=8).astype(int)) * 120
        v1 = [8] * 960
        train_index = list(map(func, train_group_label, v1, u1))
        # 测试序号
        u2 = list(np.linspace(1, 8, num=8).astype(int)) * 30
        v2 = [8] * 240
        test_index = list(map(func, test_group_label, v2, u2))

        solver = Solver(config, folder_path[config.dataset_style], folder_path[config.dataset_content], folder_path[config.dataset_stylized],train_index, test_index,Round_num)
        sr_srcc_all[p], sr_krcc_all[p], sr_plcc_all[p], sr_rmse_all[p]= solver.train()

    srcc_SR_med = np.mean(sr_srcc_all)
    krcc_SR_med = np.mean(sr_krcc_all)
    plcc_SR_med = np.mean(sr_plcc_all)
    rmse_SR_med = np.mean(sr_rmse_all)

    print('Testing SR_SRCC %4.4f,\tSR_krcc %4.4f,\tSR_PLCC %4.4f,\tSR_RMSE %4.4f' % (srcc_SR_med, krcc_SR_med,plcc_SR_med, rmse_SR_med))


    # return srcc_med, krcc_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_style', dest='dataset_style', type=str, default='style', help='path to style dataset')
    parser.add_argument('--dataset_content', dest='dataset_content', type=str, default='content', help='path to content dataset')
    parser.add_argument('--dataset_stylized', dest='dataset_stylized', type=str, default='stylized', help='path to stylized dataset')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')

    config = parser.parse_args()
    main(config)

