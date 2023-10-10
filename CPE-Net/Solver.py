import torch
from scipy import stats
import numpy as np
import models
import data_loader
from visdom import Visdom
import time
import os.path
from tqdm import tqdm

class Solver(object):

    def __init__(self, config, style_path, content_path, stylized_path, train_idx, test_idx,Round_num):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model_content = models.CPE_Net().cuda()
        self.model_content.train(True)
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.Round_num=Round_num

        content_backbone_params = list(map(id, self.model_content.res_content.parameters()))
        self.QC_params = filter(lambda p: id(p) not in content_backbone_params, self.model_content.parameters())

        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        paras_content = [{'params': self.QC_params, 'lr': self.lr * self.lrratio},
                         {'params': self.model_content.res_content.parameters(), 'lr': self.lr},
                         ]
        self.solver_content = torch.optim.Adam(paras_content, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(style_path, content_path, stylized_path,  train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(style_path, content_path, stylized_path,  test_idx, config.patch_size, config.train_patch_num, istrain=False)

        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()



    def train(self):
        """Training"""
        best_CP_srcc = 0.0
        best_CP_krcc = 0.0

        # viz = Visdom()
        # print('Epoch\tTrain_Loss\tTrain_OV_SRCC\tTest_OV_SRCC\tTest_OV_krcc\tTrain_SR_SRCC\tTest_SR_SRCC\tTest_SR_krcc\tTrain_CP_SRCC\tTest_CP_SRCC\tTest_CP_krcc')
        print('Epoch\tTrain_Loss\t\tCP_SRCC\t\tCP_krcc\t\tCP_PLCC\t\tCP_RMSE \t\ttime(min)')
        for t in range(self.epochs):
            start = time.time()
            epoch_loss = []
            pred_CP_scores = []
            gt_CP_scores = []

            for idx, (S_img, C_img, O_img, S_label, C_label, O_label) in enumerate(tqdm(self.train_data)):

                C_img = torch.as_tensor(C_img.cuda())
                O_img = torch.as_tensor(O_img.cuda())
                C_label = torch.as_tensor(C_label.cuda())
                Q_content, _ =self.model_content(C_img, O_img)
                pred_CP_scores = pred_CP_scores + Q_content.cpu().tolist()
                gt_CP_scores = gt_CP_scores + C_label.cpu().tolist()
                loss_content = self.l1_loss(Q_content.squeeze(), C_label.float().detach())
                self.solver_content.zero_grad()
                loss_content.backward()
                epoch_loss.append(loss_content.item())
                self.solver_content.step()
            test_CP_srcc, test_CP_krcc,test_CP_plcc, test_CP_rmse= self.test(self.test_data)

            if test_CP_srcc > best_CP_srcc:
                best_CP_srcc = test_CP_srcc
                best_CP_krcc = test_CP_krcc
                best_CP_plcc = test_CP_plcc
                best_CP_rmse = test_CP_rmse

            one_epoch_time= ((time.time()-start)/60)
            print(' %d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), test_CP_srcc, test_CP_krcc,test_CP_plcc, test_CP_rmse, one_epoch_time))
            '''可视化'''
            state={'model':self.model_content.state_dict(),'optimizer':self.solver_content.state_dict(),'epoch':t+1}
            torch.save(state, os.path.join('./','Round{}_epochs{}_srcc_{}.pth'.format(self.Round_num,t+1,test_CP_srcc)))
            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1

            paras_content = [{'params': self.QC_params, 'lr': self.lr * self.lrratio},
                             {'params': self.model_content.res_content.parameters(), 'lr': self.lr},
                             ]
            self.solver_content = torch.optim.Adam(paras_content, weight_decay=self.weight_decay)

        print('Best SR SRCC %f, krcc %f, plcc %f, rmse %f' %(best_CP_srcc, best_CP_krcc,best_CP_plcc, best_CP_rmse))

        return best_CP_srcc, best_CP_krcc,best_CP_plcc, best_CP_rmse

    def test(self, data):
        """Testing"""
        self.model_content.train(False)
        pred_CP_scores = []
        gt_CP_scores = []

        test_CP_srcc = np.zeros(30, dtype=np.float)
        test_CP_krcc = np.zeros(30, dtype=np.float)
        test_CP_plcc = np.zeros(30, dtype=np.float)
        test_CP_rmse = np.zeros(30, dtype=np.float)
        for idx, (S_img, C_img, O_img, S_label, C_label, O_label) in enumerate(tqdm(data)):
            C_img = torch.as_tensor(C_img.cuda())
            O_img = torch.as_tensor(O_img.cuda())
            Q_content, _ = self.model_content(C_img, O_img)
            pred_CP_scores.append(float(Q_content.item()))
            gt_CP_scores = gt_CP_scores + C_label.cpu().tolist()

        pred_CP_scores = np.mean(np.reshape(np.array(pred_CP_scores), (-1, self.test_patch_num)), axis=1)
        gt_CP_scores = np.mean(np.reshape(np.array(gt_CP_scores), (-1, self.test_patch_num)), axis=1)


        pred_CP_scores = np.reshape(np.array(pred_CP_scores),(30,8))
        gt_CP_scores = np.reshape(np.array(gt_CP_scores),(30,8))

        for i in range(30):

            test_CP_srcc[i], _ = stats.spearmanr(pred_CP_scores[i,:], gt_CP_scores[i,:])
            test_CP_krcc[i], _ = stats.kendalltau(pred_CP_scores[i,:], gt_CP_scores[i,:])
            test_CP_plcc[i], _ = stats.pearsonr(pred_CP_scores[i,:], gt_CP_scores[i,:])
            test_CP_rmse[i]= np.sqrt(((pred_CP_scores[i,:] - gt_CP_scores[i,:])**2).mean())

        srcc_CP_med = np.mean(test_CP_srcc)
        krcc_CP_med = np.mean(test_CP_krcc)
        plcc_CP_med = np.mean(test_CP_plcc)
        rmse_CP_med = np.mean(test_CP_rmse)

        self.model_content.train(True)
        return srcc_CP_med, krcc_CP_med,plcc_CP_med,rmse_CP_med
