# %load q01_pipeline/build.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')
y = bank['y']
X = bank.drop(['y'], axis=1)

columns_to_encode = ['job','marital','education','default','housing','loan','contact','day','month','poutcome']
lbe = LabelEncoder()
for column  in columns_to_encode:
    X[column] = lbe.fit_transform(X[column])
y = lbe.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=9)
model = [RandomForestClassifier()]


# Write your solution here :
def pipeline(X_train1,X_test1,y_trai1n,y_test1,model1):
    bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')
    y = bank['y']
    X = bank.drop(['y'], axis=1)
    columns_to_encode = ['job','marital','education','default','housing','loan','contact','day','month','poutcome']
    lbe = LabelEncoder()
    for column  in columns_to_encode:
        X[column] = lbe.fit_transform(X[column])
    y = lbe.fit_transform(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=9)
    smote = SMOTE(random_state=9, kind='borderline2')
    X_sampled_train,y_sampled_train = smote.fit_sample(X_train,y_train)
    rfc = RandomForestClassifier(class_weight={0:1,1:2},random_state=9)
    param_grid={
        'criterion' : ['gini','entropy'],
        'n_estimators' : [50,75,100],
        'max_depth' : [6,8,10],
        'min_samples_leaf' : [10,15,20]
    }
    grid_search =GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5)
    grid_search.fit(X_sampled_train,y_sampled_train)
    y_pred = grid_search.predict(X_test)
    return grid_search,roc_auc_score(y_test,y_pred)


