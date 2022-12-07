# -*- coding: utf-8 -*-

from imblearn.combine import SMOTETomek
import os
from imblearn.combine import *
from imblearn.over_sampling import SMOTENC
from ctgan import CTGANSynthesizer


def train_smote_tomek(train, ratio='auto'):
    """
    Args:
        train (*.csv): input train dataset
        ratio = default ='auto'
    Returns:
        _type_: pandas type 
    """
    x_train = train.drop(['Credit_Score'],axis=1)
    y_train = train['Credit_Score']

    smt = SMOTETomek(smote=None,
                    tomek=None,
                    sampling_strategy=ratio,
                    random_state=42)

    x_smt, y_smt = smt.fit_resample(x_train, y_train)
    sampling_output = x_smt.join(y_smt)
    return sampling_output



def train_smote_nc(train, ratio='auto'):
    """
    Args:
        train (*.csv): input train dataset of pandas type
        ratio = default ='auto'
    Returns:
        _type_: pandas type 
    """

    cat_col = ['Customer_ID', 'Month', 'Credit_Mix', 'Credit_History_Age', 
            'Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour']
    cat_dims = {}
    for col in cat_col:
        cat_dims[col] = len(list(train[col].unique()))
    cat_col_idx = [list(train.columns).index(col) for col in cat_col]

    
    x_train = train.drop(['Credit_Score'],axis=1)
    y_train = train['Credit_Score']


    smote_clf= SMOTENC(random_state=42, categorical_features= cat_col_idx, sampling_strategy=ratio)
    
    x_smt, y_smt = smote_clf.fit_resample(x_train, y_train)
    sampling_output = x_smt.join(y_smt)
    return sampling_output

def train_gan(train):
    cat_cols = ['Customer_ID', 'Month', 'Credit_Mix', 'Credit_History_Age', 
           'Occupation', 'Payment_of_Min_Amount', 'Payment_Behaviour','Credit_Score']
    cat_dims = {}
    for col in cat_cols:
        cat_dims[col] = len(list(train[col].unique()))
    all_col_list = [col for col in train.columns ]
    
    df_x_train = train.loc[:, all_col_list]
    
    # Pandas 
    x_train = df_x_train

    train_df_bkp = x_train.copy()
    
    ctgan = CTGANSynthesizer()
    ctgan.fit(train_df_bkp, discrete_columns=cat_cols)
    
    gan_output = ctgan.sample(100000)
    
    return gan_output
    