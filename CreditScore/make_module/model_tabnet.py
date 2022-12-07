"""
    Subject: TABNET modeling

    Status: Final
    Version:0.95
    Python Version: 3.6.8

"""
from collections import Counter
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")


class ModelTabnet:
    def __init__(
            self,
            dfs_train,  # sampled data ver.
            dfs_test,
            dfs_sampled=None,
    ):
        self.dfs_train = dfs_train
        self.dfs_test = dfs_test
        self.dfs_sampled = dfs_sampled
        self.dfs = pd.concat([self.dfs_train, self.dfs_test], ignore_index=True)

        # 1. preprocessing
        le = LabelEncoder()
        self.dfs['Month'] = le.fit_transform(self.dfs['Month'])
        self.dfs_train['Month'] = le.transform(self.dfs_train['Month'])
        self.dfs_test['Month'] = le.transform(self.dfs_test['Month'])
        if self.dfs_sampled is not None:
            self.dfs_sampled['Month'] = le.transform(self.dfs_sampled['Month'])

        # 2. Set Columns
        target = 'Credit_Score'
        cat_col = ['Customer_ID', 'Month', 'Credit_Mix', 'Credit_History_Age',
                   'Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour']
        cat_dims = {}
        for col in cat_col:
            cat_dims[col] = len(list(self.dfs[col].unique()))
            print(col, ": ", cat_dims[col])
        cat_col_idx = [list(self.dfs.columns).index(col) for col in cat_col]
        cat_col_dims = [cat_dims[col] for col in cat_col]
        all_col_list = [col for col in self.dfs.columns if col != target]

        # 3. split x/y
        self.x_train = self.dfs_train.loc[:, all_col_list].values
        self.y_train = self.dfs_train.loc[:, target].values
        self.x_test = self.dfs_test.loc[:, all_col_list].values
        self.y_test = self.dfs_test.loc[:, target].values
        if self.dfs_sampled is not None:
            self.x_sampled = self.dfs_sampled.loc[:, all_col_list].values
            self.y_sampled = self.dfs_sampled.loc[:, target].values
        else:
            self.x_sampled, self.y_sampled = None, None

        # 4. modeling
        print('!!!Pretrain Start!!!')
        pretrained_model = self.pretrain_model(cat_col_idx, cat_col_dims)
        print('!!!Main train Start!!!')
        self.model_tabnet = self.model(pretrained_model, cat_col_idx, cat_col_dims)

        # 5. predict
        self.predicted_proba, self.predict = self.make_result()

        print('*** Success TABNET Modeling Process ***')
        print('Have a good Time :)')

    def pretrain_model(self, cat_col_idx, cat_col_dims):
        x_train = self.x_train if self.x_sampled is None else self.x_sampled
        unsupervised_model = TabNetPretrainer(
            cat_idxs=cat_col_idx,
            cat_dims=cat_col_dims,
            cat_emb_dim=[int(round(np.sqrt(i)/4, 0))+1 for i in cat_col_dims],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=5*1e-3),
            scheduler_params={'is_batch_level': True, 'T_0': 10, 'T_mult': 2, 'eta_min': 0.001},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            mask_type='entmax',
        )

        unsupervised_model.fit(
            X_train=x_train,
            max_epochs=90 if self.x_sampled is None else 80,
            drop_last=False,
            pretraining_ratio=0.6
        )

        reconstructed_x, embedded_x = unsupervised_model.predict(x_train)
        assert(reconstructed_x.shape == embedded_x.shape)
        return unsupervised_model

    def model(self, pretrained_model, cat_col_idx, cat_col_dims):
        model = TabNetClassifier(
            cat_idxs=cat_col_idx,
            cat_dims=cat_col_dims,
            cat_emb_dim=[int(round(np.sqrt(i)/4, 0))+1 for i in cat_col_dims],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3),
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_params={'is_batch_level': True, 'T_0': 10, 'T_mult': 2, 'eta_min': 0.001},
            mask_type='sparsemax',
            gamma=1.3
        )

        model.fit(
            X_train=self.x_train,
            y_train=self.y_train,
            eval_set=[(self.x_train, self.y_train)],
            eval_name=['train'],
            eval_metric=['balanced_accuracy', 'accuracy'],
            max_epochs=50,
            patience=5,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            weights=1,  # cost sensitive
            from_unsupervised=pretrained_model,
        )

        return model

    def make_result(self):
        predicted_proba = pd.DataFrame()
        predict_result = pd.DataFrame()

        predicted_proba['ID'] = range(0, len(self.dfs_test))
        predicted_proba['model'] = "TABNET"
        predicted_proba['Poor'] = self.model_tabnet.predict_proba(self.x_test)[:, 0]  # final file
        predicted_proba['Standard'] = self.model_tabnet.predict_proba(self.x_test)[:, 1]
        predicted_proba['Good'] = self.model_tabnet.predict_proba(self.x_test)[:, 2]

        predict_result['ID'] = range(0, len(self.dfs_test))
        predict_result['predict'] = self.model_tabnet.predict(self.x_test)

        return predicted_proba, predict_result
