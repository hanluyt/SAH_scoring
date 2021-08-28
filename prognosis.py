# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 21:38:35 2021
@author: hanluyt
mailto: hlu20@fudan.edu.cn
"""

from grid_search import *
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import joblib

# All patients & Patients with good-grade aSAH
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--dataset', type=str, default='sah', metavar='N',
                    help='the dataset used for prognosis analysis (default: sah)')

def load_dataset(dataset_name):
    train_data = pd.read_csv(f'data/{dataset_name}_train.csv')
    test_data = pd.read_csv(f'data/{dataset_name}_test.csv')

    return train_data, test_data

# randomly down-sampling the patients with favorable outcomes in the training set
def SAH_imbalance(train_x, train_y, down_ratio=1):
    train_data = pd.concat([train_x, train_y], axis=1)
    data_maj = train_data[train_data.label == 0.]
    data_min = train_data[train_data.label == 1.]
    data_maj_sample = resample(data_maj, replace=False, n_samples=data_min.shape[0] * down_ratio)
    data = pd.concat([data_maj_sample, data_min])
    train_x, train_y = data.drop('label', axis=1), data.label
    return train_x, train_y

def get_preds(threshold, probabilities):
    return [1 if prob > threshold else 0 for prob in probabilities]

def get_fpr_tpr(y, prob):
    roc_values = []
    for thresh in np.linspace(0, 1, 50):
        preds = get_preds(thresh, prob)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        roc_values.append([tpr, fpr])
    tpr_values, fpr_values = zip(*roc_values)
    tpr_values = np.array(tpr_values)
    fpr_values = np.array(fpr_values)

    return tpr_values, fpr_values

if __name__ == "__main__":
    args = parser.parse_args()
    train_data, test_data = load_dataset(args.dataset)  # default: sah
    column_name = train_data.columns.tolist()

    print("################### traditional method #################")
    tra_auc_w, tra_recall, fpr_val_w, tpr_val_w = tradition(test_data, metric='WFNS grade')
    print('auc_w:', tra_auc_w, 'recall_w:', tra_recall)
    tra_auc_h, tra_recall, fpr_val_h, tpr_val_h = tradition(test_data, metric='Hunt-Hess grade')
    print('auc_h:', tra_auc_h, 'recall_h:', tra_recall)
    tra_auc_f, tra_recall, fpr_val_f, tpr_val_f = tradition(test_data, metric='Modified Fisher grade')
    print('auc_f:', tra_auc_f, 'recall_f:', tra_recall)

    sah_standard(train_data, test_data)
    train_x, train_y = train_data.drop('label', axis=1), train_data.label
    test_x, test_y = test_data.drop('label', axis=1), test_data.label

    base_rf_ = RandomForestClassifier(class_weight="balanced")
    estimators_rf = []
    n_estimators = 11
    rf_imp = np.zeros((n_estimators, 22))

    for est in range(n_estimators):
        X, y = SAH_imbalance(train_x, train_y)
        # bagging
        tuned_estimators = [{'n_estimators': [100, 200, 300]}]
        clf_rf = GridSearchCV(base_rf_, tuned_estimators, scoring="roc_auc")
        cv_estimator = sklearn.base.clone(clf_rf).fit(X, y)
        estimators_rf.append(cv_estimator)
        estimator = cv_estimator.best_estimator_

        rf_imp[est] = estimator.feature_importances_

    rf_imp = rf_imp.mean(axis=0)
    print('rf imp:', rf_imp)
    print('------------------')

    y_pred_rf = np.array([model.predict_proba(test_x)[:, 1] for model in estimators_rf]).mean(axis=0)
    rf_pred = [1 if ele > 0.5 else 0 for ele in y_pred_rf.tolist()]
    rf_auc, rf_acc, rf_recall = roc_auc_score(test_y, y_pred_rf), accuracy_score(test_y, rf_pred), \
                                    recall_score(test_y, rf_pred, average=None)

    print("rf_auc:", rf_auc, "rf spe:", rf_recall[0], "rf sen:", rf_recall[1])
    joblib.dump(estimators_rf, 'rf.pkl')
















