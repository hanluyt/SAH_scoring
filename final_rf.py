import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

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

# The best model for this problem: random forest
def rf_cv(train_x, train_y, test_x, test_y):
    # Set the parameters by cross-validation (default:5-fold)
    tuned_rf = [{'n_estimators': [100, 200, 300, 400]}]
    rf_clf = GridSearchCV(RandomForestClassifier(class_weight="balanced"),
                          tuned_rf, scoring="accuracy")
    rf_clf.fit(train_x, train_y)
    y_true, y_pred = test_y, rf_clf.predict(test_x)
    auc = roc_auc_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    base = rf_clf.best_estimator_
    base.fit(train_x, train_y)
    rf_imp = base.feature_importances_
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)
    dump(rf_clf, 'rf_final.joblib')
    return auc, recall, acc, rf_imp

if __name__ == '__main__':

    test_x, test_y = SAH_data("testing_set_dum.csv")
    train_x, train_y = SAH_data("training_set_dum.csv", balance=True)
    auc, recall, acc, imp = rf_cv(train_x, train_y, test_x, test_y)
    print("auc=", auc, "acc=", acc, "recall=", recall)

