from sklearn.metrics import roc_curve,auc,roc_auc_score ,accuracy_score, precision_recall_curve, average_precision_score, f1_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
class Compare_Model:
    def __init__(self , proba_list , pred_list, test):
        self.proba_list = proba_list
        self.pred_list = pred_list
        self.y_test = test['Credit_Score']

        self.pred_name = ['CB','RF','TN','HV','SV']
        self.proba_name = ['CB','RF','TN','SV']
        plt.figure(figsize=(20, 5))

        self.draw_f1()
        self.draw_auc()
        self.draw_roc_ovr()
        self.draw_roc_ovo()
        self.draw_prc_ap()
        plt.show()

    def draw_f1(self):
        f1_list = []
        for pred in self.pred_list:
            f1_list.append(f1_score(self.y_test, pred['predict'], average='macro'))
        plt.subplot(1, 5, 1)

        plt.title('f1_score')

        # bar plot
        dist = (max(f1_list) - min(f1_list)) / 2
        plt.ylim([min(f1_list) - dist, max(f1_list) + dist])
        plt.bar(self.pred_name, f1_list, color = ['pink', 'lightgreen', 'coral', 'lightgray', 'skyblue'])

        # label 삽입
        for num, height in enumerate(f1_list):
            if height <= 1:
                plt.text(self.pred_name[num], height, '%.3f' % height, ha='center', va='bottom', size=12)
            else:
                plt.text(self.pred_name[num], height, '%.3d' % height, ha='center', va='bottom', size=12)

    def draw_auc(self):
        auc_list = []
        for pred in self.pred_list:
            auc_list.append(accuracy_score(self.y_test, pred['predict']))
        plt.subplot(1, 5, 2)

        plt.title('accuracy')

        # bar plot
        dist = (max(auc_list) - min(auc_list)) / 2
        plt.ylim([min(auc_list) - dist, max(auc_list) + dist])
        plt.bar(self.pred_name, auc_list, color=['pink', 'lightgreen', 'coral', 'lightgray', 'skyblue'])

        # label 삽입
        for num, height in enumerate(auc_list):
            if height <= 1:
                plt.text(self.pred_name[num], height, '%.3f' % height, ha='center', va='bottom', size=12)
            else:
                plt.text(self.pred_name[num], height, '%.3d' % height, ha='center', va='bottom', size=12)


    def draw_roc_ovr(self):
        roc_list = []
        for proba in self.proba_list:
            roc_list.append(roc_auc_score(self.y_test, proba.iloc[:,-3:].values, multi_class='ovr'))
        plt.subplot(1, 5, 3)

        plt.title('auroc_ovr')

        # bar plot
        dist = (max(roc_list) - min(roc_list)) / 2
        plt.ylim([min(roc_list) - dist, max(roc_list) + dist])
        plt.bar(self.proba_name, roc_list, color=['pink', 'lightgreen', 'coral', 'skyblue'])

        # label 삽입
        for num, height in enumerate(roc_list):
            if height <= 1:
                plt.text(self.proba_name[num], height, '%.3f' % height, ha='center', va='bottom', size=12)
            else:
                plt.text(self.proba_name[num], height, '%.3d' % height, ha='center', va='bottom', size=12)

    def draw_roc_ovo(self):
        roc_list = []
        for proba in self.proba_list:
            roc_list.append(roc_auc_score(self.y_test, proba.iloc[:, -3:].values, multi_class='ovo'))
        plt.subplot(1, 5, 4)

        plt.title('auroc_ovo')

        # bar plot
        dist = (max(roc_list) - min(roc_list)) / 2
        plt.ylim([min(roc_list) - dist, max(roc_list) + dist])
        plt.bar(self.proba_name, roc_list, color=['pink', 'lightgreen', 'coral', 'skyblue'])

        # label 삽입
        for num, height in enumerate(roc_list):
            if height <= 1:
                plt.text(self.proba_name[num], height, '%.3f' % height, ha='center', va='bottom', size=12)
            else:
                plt.text(self.proba_name[num], height, '%.3d' % height, ha='center', va='bottom', size=12)

    def draw_prc_ap(self):
        pr_list = []
        for proba in self.proba_list:
            pr_list.append(self.auprc_score( proba.iloc[:,-3:].values))
        plt.subplot(1, 5, 5)

        plt.title('auprc_ap')

        # bar plot
        dist = (max(pr_list) - min(pr_list)) / 2
        plt.ylim([min(pr_list) - dist, max(pr_list) + dist])
        plt.bar(self.proba_name, pr_list, color=['pink', 'lightgreen', 'coral', 'skyblue'])

        # label 삽입
        for num, height in enumerate(pr_list):
            if height <= 1:
                plt.text(self.proba_name[num], height, '%.3f' % height, ha='center', va='bottom', size=12)
            else:
                plt.text(self.proba_name[num], height, '%.3d' % height, ha='center', va='bottom', size=12)

    def auprc_score(self, y_pred):
        precision = dict()
        recall = dict()
        average_precision = dict()
        y_test_dummies = pd.get_dummies(self.y_test).values

        for i in range(np.unique(self.y_test).__len__()):
            precision[i], recall[i], _ = precision_recall_curve(y_test_dummies[:, i], y_pred[:, i])
            average_precision[i] = average_precision_score(y_test_dummies[:, i], y_pred[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test_dummies.ravel(), y_pred.ravel()
        )
        average_precision["micro"] = average_precision_score(y_test_dummies, y_pred, average="micro")
        return average_precision["micro"]

class ROC_Curve:
    def __init__(self,pred_list,y_test):
        self.pred_list = pred_list
        self.y_test = y_test['Credit_Score']
        self.draw()
        plt.show()
    def draw(self):
        y_test_dummies = pd.get_dummies(self.y_test).values
        proba_name = ['CB','RF','TN','SV']
        plt.figure(figsize=(20, 5))

        for idx, m in  enumerate(self.pred_list):

            plt.subplot(1, 4, idx+1)
            plt.title(proba_name[idx])

            n_classes = 3
            class_name = ['Poor', 'Standard', 'Good']
            color = ['pink', 'lightgray', 'skyblue']

            for i in range(n_classes):
                if idx == 3:
                    FalsePositiveRate, TruePositiveRate, threshold = roc_curve(y_test_dummies[:, i],
                                                                           m.values[:, i + 1])
                else:
                    FalsePositiveRate, TruePositiveRate, threshold = roc_curve(y_test_dummies[:, i],
                                                                          m.values[:, i+2])
                area = auc(FalsePositiveRate, TruePositiveRate)


                plt.plot(
                    np.array(FalsePositiveRate),
                    np.array(TruePositiveRate),
                    color=color[i],
                    label=f'{class_name[i]} roc curve (area {np.round(area, 3)})')

                plt.fill_between(np.array(FalsePositiveRate), np.array(TruePositiveRate), alpha=0.5, color=color[i])
                plt.xlabel('FalsePositiveRate')
                plt.ylabel('TruePositiveRate')
                plt.legend()



class PR_Curve:
    def __init__(self, pred_list, y_test):
        self.pred_list = pred_list
        self.y_test = y_test['Credit_Score']
        self.draw()
        plt.show()
    def draw(self):
        y_test_dummies = pd.get_dummies(self.y_test).values
        proba_name = ['CB','RF','TN','SV']
        plt.figure(figsize=(20, 5))
        for idx, m in enumerate(self.pred_list):
            plt.subplot(1, 4, idx + 1)
            plt.title(proba_name[idx])

            n_classes = 3
            class_name = ['Poor', 'Standard', 'Good']
            color = ['pink', 'lightgray', 'skyblue']

            precision = {}
            recall = {}
            average_precision = {}

            for i in range(n_classes):
                # metric 계산
                if idx == 3:
                    precision[i], recall[i], threshold = precision_recall_curve(y_test_dummies[:, i], m.values[:, i+1])
                    average_precision[i] = average_precision_score(y_test_dummies[:, i], m.values[:, i+1])
                else:
                    precision[i], recall[i], threshold = precision_recall_curve(y_test_dummies[:, i],
                                                                                m.values[:, i + 2])
                    average_precision[i] = average_precision_score(y_test_dummies[:, i], m.values[:, i + 2])
                area = auc(recall[i], precision[i])

                plt.plot(np.array(recall[i]),
                         np.array(precision[i]),
                         color=color[i],
                         label=f'{ class_name[i] } PR curve (area = {np.round(area, 3) })')

                plt.fill_between(np.array(recall[i]), np.array(precision[i]), alpha=0.5, color=color[i])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()


