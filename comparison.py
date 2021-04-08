import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

ESTIMATOR = 7 # base classifier for svm and lr
def SAH_data(filename, balance=False):
    # Load data and handle imbalanced classes by up-sample minority class
    data = pd.read_csv(filename)
    data = data.drop(['DSA'], axis=1)
    if balance:

        data_maj = data[data.mRS == 0]
        data_min = data[data.mRS == 1]
        data_min_sample = resample(data_min,
                                   replace=True,
                                   n_samples=data_maj.shape[0])
        data = pd.concat([data_maj, data_min_sample])

    y = data.mRS
    X = data.drop('mRS', axis=1)
    return X, y

# classification according to threshold
def get_preds(threshold, probabilities):
    return [1 if prob > threshold else 0 for prob in probabilities]

def get_auc(prob, test_y):
    roc_values = []
    for thresh in np.linspace(0, 1, 40):
        preds = get_preds(thresh, prob)
        tn, fp, fn, tp = confusion_matrix(test_y, preds).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        roc_values.append([tpr, fpr])
    tpr_values, fpr_values = zip(*roc_values)
    tpr_values = np.array(tpr_values)
    fpr_values = np.array(fpr_values)
    roc_auc = auc(fpr_values, tpr_values)
    return tpr_values, fpr_values, roc_auc

def svm_ensemble(train_x, train_y, test_x, test_y):
    # Set the parameters by cross-validation (default:5-fold)
    tuned_svm = [{'C': [1, 10, 100, 1000]}]
    svm_clf = GridSearchCV(svm.SVC(kernel="linear", probability=True, class_weight="balanced"),
                           tuned_svm, scoring="accuracy")
    svm_bagging = BaggingClassifier(svm_clf, n_estimators=ESTIMATOR, max_samples=0.2)
    svm_bagging.fit(train_x, train_y)
    probas = svm_bagging.predict_proba(test_x)[:, 1]
    tpr_values, fpr_values, roc_auc = get_auc(probas, test_y)
    y_true, y_pred = test_y, svm_bagging.predict(test_x)
    recall = recall_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    svm_imp = np.zeros((ESTIMATOR, 22))
    for i in range(ESTIMATOR):
        grid_base = svm_bagging.estimators_[i]
        base = grid_base.best_estimator_
        base.fit(train_x, train_y)
        svm_imp[i] = abs(base.coef_)
    svm_imp = np.mean(svm_imp, axis=0)
    return tpr_values, fpr_values, roc_auc, acc, recall, svm_imp

def lr_emsemble(train_x, train_y, test_x, test_y):
    # Set the parameters by cross-validation (default:5-fold)
    tuned_lr = [{'C': [1, 10, 100, 1000]}]
    lr_clf = GridSearchCV(LogisticRegression(max_iter=10000, class_weight="balanced"), tuned_lr, scoring="accuracy")
    lr_bagging = BaggingClassifier(lr_clf, n_estimators=ESTIMATOR, max_samples=0.2)
    lr_bagging.fit(train_x, train_y)
    probas = lr_bagging.predict_proba(test_x)[:, 1]
    tpr_values, fpr_values, roc_auc = get_auc(probas, test_y)
    y_true, y_pred = test_y, lr_bagging.predict(test_x)
    recall = recall_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    lr_imp = np.zeros((ESTIMATOR, 22))
    for i in range(ESTIMATOR):
        grid_base = lr_bagging.estimators_[i]
        base = grid_base.best_estimator_
        base.fit(train_x, train_y)
        lr_imp[i] = abs(base.coef_)
    lr_imp = np.mean(lr_imp, axis=0)
    return tpr_values, fpr_values, roc_auc, acc, recall, lr_imp

def rf_emsemble(train_x, train_y, test_x, test_y):
    # Set the parameters by cross-validation (default:5-fold)
    tuned_rf = [{'n_estimators': [100, 200, 300, 400]}]
    rf_clf = GridSearchCV(RandomForestClassifier(class_weight="balanced"),
                          tuned_rf, scoring="accuracy")
    rf_clf.fit(train_x, train_y)
    probas = rf_clf.predict_proba(test_x)[:, 1]
    tpr_values, fpr_values, roc_auc = get_auc(probas, test_y)
    y_true, y_pred = test_y, rf_clf.predict(test_x)
    recall = recall_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    base = rf_clf.best_estimator_
    base.fit(train_x, train_y)
    rf_imp = base.feature_importances_
    return tpr_values, fpr_values, roc_auc, acc, recall, rf_imp

