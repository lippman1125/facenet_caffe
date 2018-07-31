import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def cal_hist_point(gt, score, thr_l=0.4, thr_r=0.5):

    p = 0
    n = 0

    gt_len = len(gt)
    score_len = len(score)

    comp_len = min(gt_len, score_len)

    for i in range(0, comp_len):
        if score[i] >= thr_l and score[i] < thr_r:
            if gt[i] == 1.0:
                p += 1
            else:
                n += 1

    print("range[{}, {}] p={}, n={}".format(thr_l, thr_r, p, n))

    return thr_l, thr_r, p, n


def cal_far_frr_point(gt, score, thr=0.5):
    # far = false acceptance rate  = false positive / inter match num
    # frr = false rejection rate   = false negative / intra match num
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    inter_num = 0
    intra_num = 0

    gt_len = len(gt)
    score_len = len(score)

    comp_len = min(gt_len, score_len)

    for i in range(0, comp_len):
        if float(gt[i]) >= 1.0:
            intra_num += 1
        if float(gt[i]) <= 0.0:
            inter_num += 1

        if float(score[i]) >= thr and float(gt[i]) >= 1.0:
            tp += 1
        elif float(score[i]) < thr and float(gt[i]) <= 0.0:
            tn += 1
        elif float(score[i]) >= thr and float(gt[i]) <= 0.0:
            fp += 1
        elif float(score[i]) < thr and float(gt[i]) >= 1.0:
            fn += 1

    frr = float(fn) / intra_num
    far = float(fp) / inter_num
    # print("thr={}, tp={}, fp={}, tn={}, fn={}".format(thr, tp, fp, tn, fn))
    # print("intra_num={}, inter_num={}".format(intra_num, inter_num))
    print("FRR={}, FAR={}".format(frr, far))

    return frr, far


def cal_roc_point(gt, score, thr=0.5):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # tpr = recall = tp / (tp + fn)
    # fpr = fp / (fp + tn)
    # fp

    gt_len = len(gt)
    score_len = len(score)

    comp_len = min(gt_len, score_len)

    for i in range(0, comp_len):
        if float(score[i]) >= thr and float(gt[i]) >= 1.0:
            tp += 1
        elif float(score[i]) < thr and float(gt[i]) <= 0.0:
            tn += 1
        elif float(score[i]) >= thr and float(gt[i]) <= 0.0:
            fp += 1
        elif float(score[i]) < thr and float(gt[i]) >= 1.0:
            fn += 1

    # print("thr={}, tp={}, fp={}, tn={}, fn={}".format(thr, tp, fp, tn, fn))


    recall = float(tp) / (tp + fn)

    print("recall={}, fp={}".format(recall, fp))

    return recall, fp

# gt_file = "face_list_lfw_gt.txt"
# score_file = "vggface2_lfwdata/facenet_vggface2_inception_resnet_v2_192k_lfw.txt"
# gt_file = "face_list_gt.txt"
# score_file = "vggface2_160data/facenet_vggface2_inception_resnet_v2_192k_160data.txt"

if __name__ == '__main__':
    gt_file = sys.argv[1]
    score_file = sys.argv[2]

    if not os.path.exists(gt_file):
        print("{} does not exist, quit".format(gt_file))
        exit()

    if not os.path.exists(score_file):
        print("{} does not exist, quit".format(score_file))
        exit()

    list_gt_fd = open(gt_file, "r")
    gt_lines = list_gt_fd.readlines()

    list_score_fd = open(score_file, "r")
    score_lines = list_score_fd.readlines()

    gt_list = []
    score_list = []

    for idx, l in enumerate(gt_lines):
        gt, lm, rm = l.strip("\n").split(" ")
        score = score_lines[idx].strip("\n")
        # print(gt, score)
        # if int(gt) == 1 and float(score) < 0.2:
        #     print("____{}_{}".format(score, idx))
        #
        # if int(gt) == 0 and float(score) > 0.5:
        #     print("####{}_{}".format(score, idx))

        gt_list.append(float(gt))
        score_list.append(float(score))


    thrs = np.arange(0, 1, 0.01)[::-1]
    print(thrs)

    # calculate roc
    recall_list = []
    fp_list = []
    for thr in thrs:
        recall, fp = cal_roc_point(gt_list, score_list, thr)
        recall_list.append(recall)
        fp_list.append(fp)

    # calculate auc
    fp_max = max(fp_list)
    fpr_list = []
    for idx in range(0, len(fp_list)):
        fpr = float(fp_list[idx])/fp_max
        fpr_list.append(fpr)
        print("fpr={}".format(fpr))

    auc = 0
    for idx in range(0, len(recall_list) - 1):
        auc += (recall_list[idx] + recall_list[idx + 1]) * (fpr_list[idx + 1] - fpr_list[idx]) / 2
        print("auc={}".format(auc))

    print("AUC={}".format(auc))

    # calculate far & frr
    frr_list = []
    far_list = []
    for thr in thrs:
        frr, far = cal_far_frr_point(gt_list, score_list, thr)
        frr_list.append(frr)
        far_list.append(far)

    thrs = np.arange(0, 1.0, 0.1)
    print(thrs)

    # calculate histgram
    n_list = []
    p_list = []
    for thr in thrs:
        thr_l, thr_r, p, n = cal_hist_point(gt_list, score_list, thr, thr + 0.1)
        n_list.append(n)
        p_list.append(p)

    # draw far & frr
    plt.figure(1)
    plt.plot(far_list, frr_list)

    x = np.arange(0.0, 1.1, 0.1)
    y = np.arange(0.0, 1.1, 0.1)
    plt.xticks(x)
    plt.yticks(y)
    plt.title('FRR-FAR-Curve')
    plt.ylabel("FRR")
    plt.xlabel("FAR")
    plt.grid(color = 'k', linestyle = ":")
    plt.title(os.path.basename(score_file))

    # draw roc
    plt.figure(2)
    plt.plot(fp_list, recall_list, label="AUC={}".format(auc))

    x = np.arange(0.0, 1100.0, 100.0)
    y = np.arange(0.0, 1.1, 0.1)
    plt.xticks(x)
    plt.yticks(y)
    plt.title('ROC-Curve')
    plt.ylabel("RECALL")
    plt.xlabel("FP")
    plt.legend(loc="center")
    plt.grid(color = 'k', linestyle = ":")
    plt.title(os.path.basename(score_file))

    # draw histogram
    plt.figure(3)
    plt.bar(left=thrs, height=n_list, width=0.1, label="negative", facecolor="r", edgecolor="w", align="edge")
    plt.bar(left=thrs, height=p_list, width=0.1, bottom=n_list, label="positive", facecolor="g", edgecolor="w", align="edge")
    plt.legend()
    plt.title('Sample Histogram')
    plt.ylabel("Num")
    plt.xlabel("Thresh")

    plt.show()