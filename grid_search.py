import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, confusion_matrix, auc
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tradition(test_data, metric='WFNS grade'):
    if metric not in ['WFNS grade', 'Hunt-Hess grade', 'Modified Fisher grade']:
        raise ValueError(f'{metric} is not supported.')
    X_test = test_data[test_data.columns.tolist()[:-1]]
    y_test = test_data['label']
    fpr_val = []
    tpr_val = []
    for thre in range(int(np.max(np.unique(X_test[metric])) + 1)):
        y_pred = [1 if ele > thre else 0 for ele in X_test[metric]]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        fpr_val.append(fpr)
        tpr_val.append(tpr)
    tra_auc = auc(np.array(fpr_val), np.array(tpr_val))
    if metric == 'WFNS grade' or 'Hunt-Hess grade':
        y_pred_b = [1 if ele > 3 else 0 for ele in X_test[metric]]
    elif metric == 'Modified Fisher grade':
        y_pred_b = [1 if ele > 2 else 0 for ele in X_test[metric]]
    # tra_acc = accuracy_score(y_test, y_pred_b)
    tra_recall = recall_score(y_test, y_pred_b, average=None)
    # tra_conf = confusion_matrix(y_test, y_pred_b)
    return tra_auc, tra_recall, fpr_val, tpr_val

def sah_standard(train_data, test_data):
    attr_list = ["WFNS grade", "Modified Fisher grade", "Hunt-Hess grade", "Aneurysm size",
           "Neck size", "Dome to neck ratio", "Age"]
    scaler_list = dict()
    for i in attr_list:
        tmp = train_data[i]
        scaler = MinMaxScaler()
        train_data[i] = scaler.fit_transform(tmp.to_numpy().reshape(-1, 1))
        scaler_list[i] = scaler
    for i in attr_list:
        tmp_test = test_data[i]
        scaler = scaler_list[i]
        test_data[i] = scaler.transform(tmp_test.to_numpy().reshape(-1, 1))

def dataset_preprocessing(train_data, valid_data, test_data):
    train_x, train_y = train_data.drop('label', axis=1), train_data.label
    valid_x, valid_y = valid_data.drop('label', axis=1), valid_data.label
    test_x, test_y = test_data.drop('label', axis=1), test_data.label
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def SAH_imbalance(train_x, train_y, down_ratio=1):
    train_data = pd.concat([train_x, train_y], axis=1)
    data_maj = train_data[train_data.label == 0.]
    data_min = train_data[train_data.label == 1.]
    data_maj_sample = resample(data_maj, replace=False, n_samples=data_min.shape[0] * down_ratio)
    data = pd.concat([data_maj_sample, data_min])
    train_x, train_y = data.drop('label', axis=1), data.label
    return train_x, train_y

class Ensemble(object):
    def __init__(self, n_estimators=11):
        self.estimators_svm = []
        self.estimators_lr = []
        self.estimators_rf = []
        self.n_estimators = n_estimators

        self.base_svm_ = svm.SVC(kernel="linear", probability=True, class_weight='balanced')
        self.base_lr_ = LogisticRegression(max_iter=10000, class_weight='balanced')
        self.base_rf_ = RandomForestClassifier(class_weight="balanced")

    def fit_data(self, train_x, train_y):
        for _ in range(self.n_estimators):
            X, y = SAH_imbalance(train_x, train_y)
            # bagging
            tuned_C = [{'C': [0.1, 1, 10, 100]}]
            tuned_estimators = [{'n_estimators': [100, 200, 300]}]
            clf_svm = GridSearchCV(self.base_svm_, tuned_C, scoring="roc_auc")
            clf_lr = GridSearchCV(self.base_lr_, tuned_C, scoring="roc_auc")
            clf_rf = GridSearchCV(self.base_rf_, tuned_estimators, scoring="roc_auc")
            self.estimators_svm.append(sklearn.base.clone(clf_svm).fit(X, y))
            self.estimators_lr.append(sklearn.base.clone(clf_lr).fit(X, y))
            self.estimators_rf.append(sklearn.base.clone(clf_rf).fit(X, y))

    def predict_proba(self, X):

        y_pred_svm = np.array([model.predict_proba(X)[:, 1] for model in self.estimators_svm]).mean(axis=0)
        y_pred_lr = np.array([model.predict_proba(X)[:, 1] for model in self.estimators_lr]).mean(axis=0)
        y_pred_rf = np.array([model.predict_proba(X)[:, 1] for model in self.estimators_rf]).mean(axis=0)

        return y_pred_svm, y_pred_lr, y_pred_rf

    def get_preds(self, threshold, probabilities):
        return [1 if prob > threshold else 0 for prob in probabilities]

    def get_fpr_tpr(self, y, prob):
        roc_values = []
        for thresh in np.linspace(0, 1, 50):
            preds = self.get_preds(thresh, prob)
            tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            roc_values.append([tpr, fpr])
        tpr_values, fpr_values = zip(*roc_values)
        tpr_values = np.array(tpr_values)
        fpr_values = np.array(fpr_values)

        return tpr_values, fpr_values

    def fpr_tpr_total(self, X, y):
        y_pred_svm, y_pred_lr, y_pred_rf = self.predict_proba(X)
        tpr_svm, fpr_svm = self.get_fpr_tpr(y, y_pred_svm)
        tpr_lr, fpr_lr = self.get_fpr_tpr(y, y_pred_lr)
        tpr_rf, fpr_rf = self.get_fpr_tpr(y, y_pred_rf)
        return tpr_svm, fpr_svm, tpr_lr, fpr_lr, tpr_rf, fpr_rf

    def score(self, y, probas, y_pred):
        return roc_auc_score(y, probas), recall_score(y, y_pred, average=None)

    def score_all(self, X, y, threshold=0.5):
        y_pred_svm, y_pred_lr, y_pred_rf = self.predict_proba(X)
        svm_pred = [1 if ele > threshold else 0 for ele in y_pred_svm.tolist()]
        lr_pred = [1 if ele > threshold else 0 for ele in y_pred_lr.tolist()]
        rf_pred = [1 if ele > threshold else 0 for ele in y_pred_rf.tolist()]
        svm_auc, svm_recall = self.score(y, y_pred_svm, svm_pred)
        lr_auc, lr_recall = self.score(y, y_pred_lr, lr_pred)
        rf_auc, rf_recall = self.score(y, y_pred_rf, rf_pred)
        return svm_auc, svm_recall, lr_auc, lr_recall, rf_auc, rf_recall



















