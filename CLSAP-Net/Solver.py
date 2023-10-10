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
    def __init__(self, config, style_path, content_path, stylized_path, train_idx, test_idx, Round_num):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.SRE_Net= models.SRE_Net().cuda()
        self.CPE_Net= models.CPE_Net().cuda()

        self.Round_num = Round_num
        # load the pretrained model
        Pre_style_model_path = './xxx.pth'
        checkpoint_style = torch.load(Pre_style_model_path)
        self.SRE_Net.load_state_dict(checkpoint_style['model'])

        Pre_content_model_path = './xxx.pth'
        checkpoint_content = torch.load(Pre_content_model_path)
        self.CPE_Net.load_state_dict(checkpoint_content['model'])

        self.OVT_Net = models.OVT_Net(128, 256, 128, 64, 32, 16, 8).cuda()
        self.OVT_Net.train(True)

        self.SRE_Net.train(True)
        style_backbone_params = list(map(id, self.SRE_Net.res_style.parameters()))
        self.QS_params = filter(lambda p: id(p) not in style_backbone_params, self.SRE_Net.parameters())

        self.CPE_Net.train(True)
        content_backbone_params = list(map(id, self.CPE_Net.res_content.parameters()))
        self.QC_params = filter(lambda p: id(p) not in content_backbone_params, self.CPE_Net.parameters())

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay

        gram_backbone_params = list(map(id, self.OVT_Net.style_gram_res.parameters()))
        self.hyper_params = filter(lambda p: id(p) not in gram_backbone_params, self.OVT_Net.parameters())

        paras_overall = [{'params': self.QS_params, 'lr': self.lr * self.lrratio},
                         {'params': self.SRE_Net.res_style.parameters(), 'lr': self.lr},
                         {'params': self.QC_params, 'lr': self.lr * self.lrratio},
                         {'params': self.CPE_Net.res_content.parameters(), 'lr': self.lr},
                         {'params': self.hyper_params, 'lr': self.lr * self.lrratio},
                         {'params': self.OVT_Net.style_gram_res.parameters(), 'lr': self.lr},
                         ]
        self.solver_overall = torch.optim.Adam(paras_overall, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(style_path, content_path, stylized_path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(style_path, content_path, stylized_path, test_idx, config.patch_size, config.train_patch_num, istrain=False)

        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()


    def train(self):
        """Training"""
        best_OV_srcc = 0.0
        best_OV_krcc = 0.0

        # print('Epoch\tTrain_Loss\tTrain_OV_SRCC\tTest_OV_SRCC\tTest_OV_krcc\tTrain_SR_SRCC\tTest_SR_SRCC\tTest_SR_krcc\tTrain_CP_SRCC\tTest_CP_SRCC\tTest_CP_krcc')
        print('Epoch\tTrain_Loss\t\tOV_SRCC\t\tOV_krcc\t\tOV_PLCC\t\tOV_RMSE \t\ttime(min)')
        for t in range(self.epochs):
            start = time.time()
            epoch_loss = []
            pred_OV_scores = []
            gt_OV_scores = []


            for idx, (S_img, C_img, O_img, S_label, C_label, O_label) in enumerate(tqdm(self.train_data)):

                S_img = torch.as_tensor(S_img.cuda())
                C_img = torch.as_tensor(C_img.cuda())
                O_img = torch.as_tensor(O_img.cuda())
                O_label = torch.as_tensor(O_label.cuda())

                Q_content, content_w_vec = self.CPE_Net(C_img, O_img)
                Q_style, style_w_vec   = self.SRE_Net(S_img, O_img)

                # Generate weights for target network
                paras = self.OVT_Net(S_img, content_w_vec,style_w_vec)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.SAWEH(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                Q_overall = model_target(
                    paras['c_target_in_vec'],paras['s_target_in_vec'], Q_content, Q_style)
                pred_OV_scores = pred_OV_scores + Q_overall.cpu().tolist()
                gt_OV_scores = gt_OV_scores + O_label.cpu().tolist()

                loss_overall = self.l1_loss(Q_overall.squeeze(), O_label.float().detach())
                self.solver_overall.zero_grad()
                epoch_loss.append(loss_overall.item())

                loss_overall.backward()
                self.solver_overall.step()

            test_OV_srcc, test_OV_krcc, test_OV_plcc, test_OV_rmse, pred_OV_scores, gt_OV_scores = self.test(self.test_data)

            if test_OV_srcc > best_OV_srcc:
                best_OV_srcc = test_OV_srcc
                best_OV_krcc = test_OV_krcc
                best_OV_plcc = test_OV_plcc
                best_OV_rmse = test_OV_rmse

            one_epoch_time= ((time.time()-start)/60)
            print(' %d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), test_OV_srcc, test_OV_krcc, test_OV_plcc, test_OV_rmse, one_epoch_time))

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1

            paras_overall = [{'params': self.QS_params, 'lr': self.lr * self.lrratio},
                             {'params': self.SRE_Net.res_style.parameters(), 'lr': self.lr},
                             {'params': self.QC_params, 'lr': self.lr * self.lrratio},
                             {'params': self.CPE_Net.res_content.parameters(), 'lr': self.lr},
                             {'params': self.hyper_params, 'lr': self.lr * self.lrratio},
                             {'params': self.OVT_Net.style_gram_res.parameters(), 'lr': self.lr},
                             ]
            self.solver_overall = torch.optim.Adam(paras_overall, weight_decay=self.weight_decay)


        print('Best OV SRCC %f, krcc %f, plcc %f, rmse %f' %(best_OV_srcc, best_OV_krcc, best_OV_plcc, best_OV_rmse))

        return best_OV_srcc, best_OV_krcc, best_OV_plcc, best_OV_rmse

    def test(self, data):
        """Testing"""

        self.OVT_Net.train(False)
        self.CPE_Net.train(False)
        self.SRE_Net.train(False)

        pred_OV_scores = []
        gt_OV_scores = []

        test_OV_srcc = np.zeros(30, dtype=np.float)
        test_OV_krcc = np.zeros(30, dtype=np.float)
        test_OV_plcc = np.zeros(30, dtype=np.float)
        test_OV_rmse = np.zeros(30, dtype=np.float)

        for idx, (S_img, C_img, O_img, S_label, C_label, O_label) in enumerate(tqdm(data)):
            # Data.
            S_img = torch.as_tensor(S_img.cuda())
            C_img = torch.as_tensor(C_img.cuda())
            O_img = torch.as_tensor(O_img.cuda())

            O_label = torch.as_tensor(O_label.cuda())

            Q_content, content_w_vec = self.CPE_Net(C_img, O_img)
            Q_style, style_w_vec = self.SRE_Net(S_img, O_img)

            # Generate weights for target network
            paras = self.OVT_Net(S_img, content_w_vec,style_w_vec)  # 'paras' contains the network weights conveyed to target network

            # Building target network
            model_target = models.SAWEH(paras).cuda()
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            Q_overall = model_target(
                paras['s_target_in_vec'], paras['c_target_in_vec'], Q_content,Q_style)  # while 'paras['target_in_vec']' is the input to target net

            pred_OV_scores.append(float(Q_overall.item()))
            gt_OV_scores = gt_OV_scores + O_label.cpu().tolist()



        pred_OV_scores = np.mean(np.reshape(np.array(pred_OV_scores), (-1, self.test_patch_num)), axis=1)  
        gt_OV_scores = np.mean(np.reshape(np.array(gt_OV_scores), (-1, self.test_patch_num)), axis=1)

        pred_OV_scores = np.reshape(np.array(pred_OV_scores),(30,8))
        gt_OV_scores = np.reshape(np.array(gt_OV_scores),(30,8))


        for i in range(30):
            test_OV_srcc[i], _ = stats.spearmanr(pred_OV_scores[i,:], gt_OV_scores[i,:])
            test_OV_krcc[i], _ = stats.kendalltau(pred_OV_scores[i,:], gt_OV_scores[i,:])
            test_OV_plcc[i], _ = stats.pearsonr(pred_OV_scores[i,:], gt_OV_scores[i,:])
            test_OV_rmse[i]= np.sqrt(((pred_OV_scores[i,:] - gt_OV_scores[i,:])**2).mean())


        srcc_OV_med = np.mean(test_OV_srcc)
        krcc_OV_med = np.mean(test_OV_krcc)
        plcc_OV_med = np.mean(test_OV_plcc)
        rmse_OV_med = np.mean(test_OV_rmse)


        self.OVT_Net.train(True)
        self.CPE_Net.train(True)
        self.SRE_Net.train(True)

        return srcc_OV_med, krcc_OV_med, plcc_OV_med, rmse_OV_med, pred_OV_scores,gt_OV_scores
