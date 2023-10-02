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
    """Solver for training and testing """
    def __init__(self, config, style_path, content_path, stylized_path, train_idx, test_idx,Round_num):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.model_style = models.SRE_Net().cuda()
        self.model_style.train(True)
        self.l1_loss = torch.nn.L1Loss().cuda()
        self.Round_num=Round_num
        style_backbone_params = list(map(id, self.model_style.res_style.parameters()))
        self.QS_params = filter(lambda p: id(p) not in style_backbone_params, self.model_style.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras_style = [{'params': self.QS_params, 'lr': self.lr * self.lrratio},
                       {'params': self.model_style.res_style.parameters(), 'lr': self.lr},
                       ]
        self.solver_style = torch.optim.Adam(paras_style, weight_decay=self.weight_decay)
        train_loader = data_loader.DataLoader(style_path, content_path, stylized_path,  train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(style_path, content_path, stylized_path,  test_idx, config.patch_size, config.train_patch_num, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_SR_srcc = 0.0
        best_SR_krcc = 0.0
        print('Epoch\tTrain_Loss\t\tSR_SRCC\t\tSR_krcc\t\tSR_PLCC\t\tSR_RMSE \t\ttime(min)')
        for t in range(self.epochs):
            start = time.time()
            epoch_loss = []
            pred_SR_scores = []
            gt_SR_scores = []

            for idx, (S_img, C_img, O_img, S_label, C_label, O_label) in enumerate(tqdm(self.train_data)):
                S_img = torch.as_tensor(S_img.cuda())
                O_img = torch.as_tensor(O_img.cuda())
                S_label = torch.as_tensor(S_label.cuda())
                Q_style, _= self.model_style(S_img, O_img)
                pred_SR_scores = pred_SR_scores + Q_style.cpu().tolist()
                gt_SR_scores = gt_SR_scores + S_label.cpu().tolist()
                loss_style = self.l1_loss(Q_style.squeeze(), S_label.float().detach())
                self.solver_style.zero_grad()
                loss_style.backward()
                epoch_loss.append(loss_style.item())
                self.solver_style.step()
            test_SR_srcc, test_SR_krcc ,test_SR_plcc, test_SR_rmse= self.test(self.test_data)

            if test_SR_srcc > best_SR_srcc:
                best_SR_srcc = test_SR_srcc
                best_SR_krcc = test_SR_krcc
                best_SR_plcc = test_SR_plcc
                best_SR_rmse = test_SR_rmse

            one_epoch_time= ((time.time()-start)/60)
            print(' %d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), test_SR_srcc, test_SR_krcc, test_SR_plcc, test_SR_rmse, one_epoch_time))
            '''可视化'''
            state={'model':self.model_style.state_dict(),'optimizer':self.solver_style.state_dict(),'epoch':t+1}
            torch.save(state, os.path.join('./','Round{}_epochs{}_srcc_{}.pth'.format(self.Round_num,t+1,test_SR_srcc)))
            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1

            paras_style = [{'params': self.QS_params, 'lr': self.lr * self.lrratio},
                           {'params': self.model_style.res_style.parameters(), 'lr': self.lr},
                           ]
            self.solver_style = torch.optim.Adam(paras_style, weight_decay=self.weight_decay)


        print('Best SR SRCC %f, krcc %f, plcc %f, rmse %f' %(best_SR_srcc, best_SR_krcc,best_SR_plcc, best_SR_rmse))

        return best_SR_srcc, best_SR_krcc, best_SR_plcc, best_SR_rmse

    def test(self, data):
        """Testing"""
        self.model_style.train(False)
        T = 0

        pred_SR_scores = []
        gt_SR_scores = []

        test_SR_srcc = np.zeros(30, dtype=np.float)
        test_SR_krcc = np.zeros(30, dtype=np.float)
        test_SR_plcc = np.zeros(30, dtype=np.float)
        test_SR_rmse = np.zeros(30, dtype=np.float)

        for idx, (S_img, C_img, O_img, S_label, C_label, O_label) in enumerate(tqdm(data)):
            # Data.
            S_img = torch.as_tensor(S_img.cuda())
            O_img = torch.as_tensor(O_img.cuda())
            Q_style, _ = self.model_style(S_img, O_img)
            pred_SR_scores.append(float(Q_style.item()))
            gt_SR_scores = gt_SR_scores + S_label.cpu().tolist()

        pred_SR_scores = np.mean(np.reshape(np.array(pred_SR_scores), (-1, self.test_patch_num)), axis=1)
        gt_SR_scores = np.mean(np.reshape(np.array(gt_SR_scores), (-1, self.test_patch_num)), axis=1)


        pred_SR_scores = np.reshape(np.array(pred_SR_scores),(30,8))
        gt_SR_scores = np.reshape(np.array(gt_SR_scores),(30,8))

        for i in range(30):
            test_SR_srcc[i], _ = stats.spearmanr(pred_SR_scores[i,:], gt_SR_scores[i,:])
            test_SR_krcc[i], _ = stats.kendalltau(pred_SR_scores[i,:], gt_SR_scores[i,:])
            test_SR_plcc[i], _ = stats.pearsonr(pred_SR_scores[i,:], gt_SR_scores[i,:])
            test_SR_rmse[i] = np.sqrt(((pred_SR_scores[i,:] - gt_SR_scores[i,:])**2).mean())


        srcc_SR_med = np.mean(test_SR_srcc)
        krcc_SR_med = np.mean(test_SR_krcc)
        plcc_SR_med = np.mean(test_SR_plcc)
        rmse_SR_med = np.mean(test_SR_rmse)

        self.model_style.train(True)
        return srcc_SR_med, krcc_SR_med, plcc_SR_med, rmse_SR_med