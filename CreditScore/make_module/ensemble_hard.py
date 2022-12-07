
class Hard_voting():
    def __init__(self, predict_proba ):
        self.predict_proba = predict_proba


        self.predict = self.h_voting()
    def proba_to_predict(self):
        predict_df = self.predict_proba.iloc[:, :2]
        predict = self.predict_proba.iloc[: , 2:].idxmax(axis =1)
        predict_df['predict'] = predict
        predict_df.loc[predict_df['predict'] == 'Poor', 'predict'] = 0
        predict_df.loc[predict_df['predict'] == 'Standard', 'predict'] = 1
        predict_df.loc[predict_df['predict'] == 'Good', 'predict'] = 2

        return predict_df

    def h_voting(self):
        h_voting = self.proba_to_predict()
        h_voting = h_voting.groupby('ID')['predict'].agg(**{
            'predict': lambda x: x.mode()[0] # 공동 1등일때 첫번째 값으로 들어가도록함
        }).reset_index()

        return h_voting
