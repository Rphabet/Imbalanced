"""
    Subject: Soft Ensemble

    Status: Final
    Version:0.95
    Python Version: 3.6.8

"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class EnsembleSoft:
    def __init__(
            self,
            tabnet_proba,
            catboost_proba,
            rf_proba,
            soft_ratio=None,
    ):
        if soft_ratio is None:
            soft_ratio = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]
        self.tabnet_proba = tabnet_proba
        self.catboost_proba = catboost_proba
        self.rf_proba = rf_proba
        self.soft_ratio = soft_ratio

        credit_score_dict = {'Poor': 0, 'Standard': 1, 'Good': 2}

        self.predict_proba = self.soft_voting()
        self.predict = pd.DataFrame()
        self.predict['ID'] = range(0, len(self.rf_proba))
        self.predict['predict'] = self.predict_proba.iloc[:, 1:].idxmax(axis=1)
        self.predict['predict'] = list(map(lambda s: credit_score_dict.get(s), self.predict['predict']))

    def soft_voting(self):
        df_result = pd.DataFrame()
        df_result['ID'] = range(0, len(self.rf_proba))
        df_result['Poor'] = self.tabnet_proba.iloc[:, 2] * self.soft_ratio[0][0] + \
            self.catboost_proba.iloc[:, 2] * self.soft_ratio[0][1] + \
            self.rf_proba.iloc[:, 2] * self.soft_ratio[0][2]
        df_result['Standard'] = self.tabnet_proba.iloc[:, 3] * self.soft_ratio[1][0] + \
            self.catboost_proba.iloc[:, 3] * self.soft_ratio[1][1] + \
            self.rf_proba.iloc[:, 3] * self.soft_ratio[1][2]
        df_result['Good'] = self.tabnet_proba.iloc[:, 4] * self.soft_ratio[2][0] + \
            self.catboost_proba.iloc[:, 4] * self.soft_ratio[2][1] + \
            self.rf_proba.iloc[:, 4] * self.soft_ratio[2][2]

        # 소수점 계산 때문에 확률의 합이 1이 되지 않는 경우가 발생하여 보정해주는 부분
        df_result['Standard'] = np.where(
            df_result['Poor']+df_result['Standard']+df_result['Good'] != 1,
            df_result['Standard'] + (1-(df_result['Poor']+df_result['Standard']+df_result['Good'])),
            df_result['Standard']
        )

        return df_result



