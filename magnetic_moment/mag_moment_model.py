import pandas as pd
import pickle

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


db = pd.read_csv("../mxene_additonal_features.csv")
db['stability_clf'] = db.apply(lambda row: row.stability<0.2, axis=1)
db=db*1
db.fillna(0, inplace=True)


X = db.drop(columns=['is_magnetic', 'spacegroup', 'magmom', 'stability', 'formula', 'stability_clf']).values
Y = db['magmom'].values
X, Y = shuffle(X, Y, random_state=0)


scaler = StandardScaler()
X_ = scaler.fit_transform(X)
xgb_reg = xgb.XGBRegressor(colsample_bytree=0.83174,
                     gamma=0.58847,
                     learning_rate=0.3388,
                     max_depth=5,
                     n_estimators=100,
                     subsample=0.899)

xgb_reg.fit(X_,Y)


# Saving the model
data = {
    'model': xgb_reg,
    'scaler': scaler
    }
with open('mag_mom_model.pkl', 'wb') as file:
    pickle.dump(data, file)
