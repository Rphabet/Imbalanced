import pandas as pd
from catboost import CatBoostClassifier
class ModelCatBoost:

    def __init__(self, train_data, test_data, choose, class_weights = {0: 1.1539455706953214,1: 0.6162978740188538,2: 1.9576520704128297}, params = {'bagging_temperature': 1.7667222733865562,
               'border_count': 190, 'depth': 14,
               'l2_leaf_reg': 100.0, 'learning_rate': 0.09356008116567356,
               'min_data_in_leaf': 4.281513544154046
               }):
        self.train_data = train_data
        self.test_data = test_data
        self.params = params
        self.class_weights = class_weights
        self.x_train, self.y_train, self.x_test, self.y_test = self.train_test_split()

        if choose == 'plain':
            self.cat_model = self.plain()
        elif choose == "optimize":
            self.cat_model = self.optimize_cat(params)
        self.predict_proba = self.pred_proba()
        self.predict = self.pred()
        self.feature_importance = self.get_importance()

    def proba_return(self,predict_test):
        proba_df = pd.DataFrame()
        proba_df['ID'] = range(0, len(predict_test) )
        proba_df['model'] = 'Cat'
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
        plain_model = CatBoostClassifier(random_state = 0 , class_weights = self.class_weights)
        plain_model.fit(self.x_train, self.y_train)
        return plain_model

        predict_test = plain_model.predict_proba(self.x_test)
        return self.proba_return(predict_test)

    def optimize_cat(self,params ):
        optimize_model = CatBoostClassifier(random_state = 0 ,class_weights = self.class_weights, **params)
        optimize_model.fit(self.x_train, self.y_train)
        return optimize_model


    def pred_proba(self):
        predict_proba_list = self.cat_model.predict_proba(self.x_test)
        return self.proba_return(predict_proba_list)

    def pred(self):
        predict_list =  self.cat_model.predict(self.x_test)
        predict_df = pd.DataFrame()
        predict_df['ID'] = range(0, len(predict_list))
        predict_df['predict'] = predict_list
        return predict_df


    def get_importance(self):
        importance = self.cat_model.get_feature_importance()
        return importance