import numpy as np
from scipy import stats

SR_label_txt = open('./dataset/OV_label.txt', 'r')
Q_style_txt=open('./dataset/Qoverall.txt', 'r')

SR_labels = []
for line in SR_label_txt:
    line = line.split('\n')  # 以换行为分割符
    words = line[0].split()  # 以空格为分隔符，包含 \n
    SR_labels.append((words[0]))
SR_label = np.array(SR_labels).astype(np.float32)

Q=[]
for line in Q_style_txt:
    line = line.split('\n')  # 以换行为分割符
    words = line[0].split()  # 以空格为分隔符，包含 \n
    Q.append((words[0]))
Q_style = np.array(Q).astype(np.float32)

gt_SR_scores = SR_label
pred_SR_scores=Q_style

test_SR_srcc = np.zeros(150, dtype=np.float)
test_SR_krcc = np.zeros(150, dtype=np.float)
test_SR_plcc = np.zeros(150, dtype=np.float)
test_SR_rmse = np.zeros(150, dtype=np.float)

pred_SR_scores = np.reshape(np.array(pred_SR_scores), (150, 8))
gt_SR_scores = np.reshape(np.array(gt_SR_scores), (150, 8))

for i in range(150):
    test_SR_srcc[i], _ = stats.spearmanr(pred_SR_scores[i, :], gt_SR_scores[i, :])
    test_SR_krcc[i], _ = stats.kendalltau(pred_SR_scores[i, :], gt_SR_scores[i, :])
    test_SR_plcc[i], _ = stats.pearsonr(pred_SR_scores[i, :], gt_SR_scores[i, :])
    test_SR_rmse[i] = np.sqrt(((pred_SR_scores[i, :] - gt_SR_scores[i, :]) ** 2).mean())

srcc_SR_med = np.mean(test_SR_srcc)
krcc_SR_med = np.mean(test_SR_krcc)
plcc_SR_med = np.mean(test_SR_plcc)
rmse_SR_med = np.mean(test_SR_rmse)

print('SRCC:',srcc_SR_med,'KRCC:',krcc_SR_med,'PLCC:',plcc_SR_med,'RMSE:',rmse_SR_med)






















#
# SR_label_txt = open('./dataset/label.txt', 'r')
# Q_style_txt=open('./dataset/pre.txt', 'r')
#
# SR_labels = []
# for line in SR_label_txt:
#     line = line.split('\n')  # 以换行为分割符
#     words = line[0].split()  # 以空格为分隔符，包含 \n
#     SR_labels.append((words[0]))
# SR_label = np.array(SR_labels).astype(np.float32)
#
# Q=[]
# for line in Q_style_txt:
#     line = line.split('\n')  # 以换行为分割符
#     words = line[0].split()  # 以空格为分隔符，包含 \n
#     Q.append((words[0]))
# Q_style = np.array(Q).astype(np.float32)
#
# gt_SR_scores = SR_label
# pred_SR_scores=Q_style
#
# test_SR_srcc = np.zeros(100, dtype=np.float)
# test_SR_krcc = np.zeros(100, dtype=np.float)
# test_SR_plcc = np.zeros(100, dtype=np.float)
# test_SR_rmse = np.zeros(100, dtype=np.float)
#
# pred_SR_scores = np.reshape(np.array(pred_SR_scores), (100, 10))
# gt_SR_scores = np.reshape(np.array(gt_SR_scores), (100, 10))
#
# for i in range(100):
#     test_SR_srcc[i], _ = stats.spearmanr(pred_SR_scores[i, :], gt_SR_scores[i, :])
#     test_SR_krcc[i], _ = stats.kendalltau(pred_SR_scores[i, :], gt_SR_scores[i, :])
#     test_SR_plcc[i], _ = stats.pearsonr(pred_SR_scores[i, :], gt_SR_scores[i, :])
#     test_SR_rmse[i] = np.sqrt(((pred_SR_scores[i, :] - gt_SR_scores[i, :]) ** 2).mean())
#
# srcc_SR_med = np.mean(test_SR_srcc)
# krcc_SR_med = np.mean(test_SR_krcc)
# plcc_SR_med = np.mean(test_SR_plcc)
# rmse_SR_med = np.mean(test_SR_rmse)
#
# print(srcc_SR_med,krcc_SR_med,plcc_SR_med,rmse_SR_med)