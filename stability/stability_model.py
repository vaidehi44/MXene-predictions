import pandas as pd
import pickle

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier


db = pd.read_csv("../mxene_additonal_features.csv")
db['stability_clf'] = db.apply(lambda row: row.stability<0.2, axis=1)
db=db*1
db.fillna(0, inplace=True)


X = db.drop(columns=['is_magnetic', 'spacegroup', 'magmom', 'stability', 'formula', 'stability_clf']).values
Y = db['stability_clf'].values

X, Y = shuffle(X, Y, random_state=0)


scaler = StandardScaler()
X_ = scaler.fit_transform(X)
rf = RandomForestClassifier(
    n_estimators=800,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    max_depth=100,
    bootstrap=True)

rf.fit(X_, Y)


# Saving the model
data = {
    'model': rf,
    'scaler': scaler
}
with open('stability_model.pkl', 'wb') as file:
    pickle.dump(data, file)
