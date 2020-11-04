import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import pickle

def train_resampled(X, y):
    # resampling
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

#####

ds = pd.read_csv('data/bmi.csv')

# sex encoding
sex_enc = {'Male':1,'Female':0}

ds['Gender'] = ds.Gender.apply(lambda x: sex_enc[x])

X,y = ds.drop('Index',axis=1),ds.Index

poly = PolynomialFeatures(2,include_bias=False)

X_enriched = pd.concat((pd.DataFrame(poly.fit_transform(X[['Height','Weight']]),columns=['Height','Weight','Height_2','Weight_2','interaction']),X[['Gender']]),axis=1)

X_train,y_train = train_resampled(X_enriched,y)

knn = KNeighborsClassifier(n_neighbors=5,
                           weights='uniform',
                           algorithm='auto')
knn.fit(X_train, y_train)

pickle.dump(knn, open('model.pkl','wb'))