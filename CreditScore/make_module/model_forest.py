import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class ModelForest:

    def __init__(self, train_data, test_data , choose, class_weight ={0: 1.1539455706953214,1: 0.6162978740188538,2: 1.9576520704128297} ,params = {'n_estimators': 400, 'max_depth': 20, 'min_samples_split':2 , 'min_samples_leaf': 2}):
        self.train_data = train_data
        self.test_data = test_data
        self.params = params
        self.class_weight = class_weight
        self.x_train , self.y_train, self.x_test, self.y_test = self.train_test_split()
        if choose == 'plain':
            self.forest_model = self.plain()
        elif choose == "optimize" :
            self.forest_model = self.optimize_forest(params)
        self.predict_proba = self.pred_proba()
        self.predict = self.pred()
        self.feature_importance = self.get_importance()

    def proba_return(self,predict_test):
        proba_df = pd.DataFrame()
        proba_df['ID'] = range(0, len(predict_test))
        proba_df['model'] = 'RF'
        proba_df['Poor'] = predict_test[:, 0]
        proba_df['Standard'] = predict_test[:, 1]
        proba_df['Good'] = predict_test[:, 2]

        return proba_df

    def train_test_split(self):
        x_train = self.train_data.drop(['Credit_Score'], axis=1)
        y_train = self.train_data['Credit_Score']
        x_test = self.test_data.drop(['Credit_Score'], axis=1)
        y_test = self.test_data['Credit_Score']
        return x_train , y_train , x_test, y_test

    def plain(self):
        plain_model = RandomForestClassifier(random_state = 0,class_weight = self.class_weight)
        plain_model.fit(self.x_train, self.y_train)
        return plain_model

    def optimize_forest(self, params ):
        optimize_model = RandomForestClassifier(random_state=0,class_weight = self.class_weight, **params)
        optimize_model.fit(self.x_train, self.y_train)
        return optimize_model

    def pred_proba(self):
        predict_proba_list = self.forest_model.predict_proba(self.x_test)

        return self.proba_return(predict_proba_list)

    def pred(self):
        predict_list =  self.forest_model.predict(self.x_test)
        predict_df = pd.DataFrame()
        predict_df['ID'] = range(0, len(predict_list))
        predict_df['predict'] = predict_list
        return predict_df


    def get_importance(self):
        importance = self.forest_model.feature_importances_
        #std = np.std([tree.feature_importances_ for tree in self.forest_model.estimators_], axis =0)
        #indices = np.argsort(importance)[::-1]
        return importance
