from grid_search import *
import pandas as pd
import numpy as np
from mesa import Mesa
from arguments import parser
from utils import Rater, load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold

# def get_preds(threshold, probabilities):
#     return [1 if prob > threshold else 0 for prob in probabilities]
#
# def get_fpr_tpr(y, prob):
#     roc_values = []
#     for thresh in np.linspace(0, 1, 50):
#         preds = get_preds(thresh, prob)
#         tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
#         tpr = tp / (tp + fn)
#         fpr = fp / (fp + tn)
#         roc_values.append([tpr, fpr])
#     tpr_values, fpr_values = zip(*roc_values)
#     tpr_values = np.array(tpr_values)
#     fpr_values = np.array(fpr_values)
#
#     return tpr_values, fpr_values

if __name__ == "__main__":
    # randomly split 10 times
    n_runs = 10

    args = parser.parse_args()
    rater = Rater(args.metric)  # default: auc

    # load data and standardization
    train_data, test_data = load_dataset(args.dataset)
    sah_standard(train_data, test_data)

    column_name = train_data.columns.tolist()
    X = train_data.drop('label', axis=1)
    y = train_data.label
    X, y = X.values, y.values

    # Four ML model
    base_estimator = DecisionTreeClassifier(max_depth=None)
    mesa = Mesa(
        args=args,
        base_estimator=base_estimator,
        n_estimators=args.max_estimators)
    down_ensemble = Ensemble(n_estimators=11)

    # 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    scores_list_train, scores_list_valid = [], []

    n = 1
    for _ in range(n_runs):
        for train_index, val_index in skf.split(X, y):
            print("cross validation (accumulate):", n)
            n += 1
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            y_train = y_train[:, np.newaxis]
            y_val = y_val[:, np.newaxis]
            train_data = np.concatenate((X_train, y_train), axis=1)
            valid_data = np.concatenate((X_val, y_val), axis=1)
            train_data = pd.DataFrame(train_data, columns=column_name)
            valid_data = pd.DataFrame(valid_data, columns=column_name)

            train_x, train_y, valid_x, valid_y, test_x, test_y = dataset_preprocessing(train_data, valid_data, test_data)

            # fit model
            down_ensemble.fit_data(train_x, train_y)
            mesa.meta_fit(train_x, train_y, valid_x, valid_y, test_x, test_y)
            mesa.fit(train_x, train_y, valid_x, valid_y, verbose=False)

            # result (AUC, spe, sen)
            mesa_auc_train = rater.score(train_y, mesa.predict_proba(train_x)[:, 1])
            y_pred_b_train = mesa.predict_proba(train_x)[:, 1]
            y_pred_b_train[y_pred_b_train < 0.5] = 0
            y_pred_b_train[y_pred_b_train >= 0.5] = 1
            mesa_recall_train = recall_score(train_y, y_pred_b_train, average=None)
            svm_auc_train, svm_recall_train, lr_auc_train, lr_recall_train, \
            rf_auc_train, rf_recall_train = down_ensemble.score_all(train_x, train_y)

            scores_list_train.append([svm_auc_train, svm_recall_train[0], svm_recall_train[1],
                                      lr_auc_train, lr_recall_train[0], lr_recall_train[1],
                                      rf_auc_train, rf_recall_train[0], rf_recall_train[1],
                                      mesa_auc_train, mesa_recall_train[0], mesa_recall_train[1]])

            mesa_auc_valid = rater.score(valid_y, mesa.predict_proba(valid_x)[:, 1])
            y_pred_b_valid = mesa.predict_proba(valid_x)[:, 1]
            y_pred_b_valid[y_pred_b_valid < 0.5] = 0
            y_pred_b_valid[y_pred_b_valid >= 0.5] = 1
            mesa_recall_valid = recall_score(valid_y, y_pred_b_valid, average=None)
            svm_auc_valid, svm_recall_valid, lr_auc_valid, lr_recall_valid, \
            rf_auc_valid, rf_recall_valid = down_ensemble.score_all(valid_x, valid_y)
            scores_list_valid.append([svm_auc_valid, svm_recall_valid[0], svm_recall_valid[1],
                                      lr_auc_valid, lr_recall_valid[0], lr_recall_valid[1],
                                      rf_auc_valid, rf_recall_valid[0], rf_recall_valid[1],
                                      mesa_auc_valid, mesa_recall_valid[0], mesa_recall_valid[1]])

    df_scores_train = pd.DataFrame(scores_list_train, columns=['svm_auc', 'svm_spe', 'svm_sen',
                                                               'lr_auc', 'lr_spe', 'lr_sen',
                                                               'rf_auc', 'rf_spe', 'rf_sen',
                                                               'mesa_auc', 'mesa_spe', 'mesa_sen'])

    df_scores_valid = pd.DataFrame(scores_list_valid, columns=['svm_auc', 'svm_spe', 'svm_sen',
                                                               'lr_auc', 'lr_spe', 'lr_sen',
                                                               'rf_auc', 'rf_spe', 'rf_sen',
                                                               'mesa_auc', 'mesa_spe', 'mesa_sen'])

    df_scores_train.to_csv('ml_train.csv', index=False)
    df_scores_valid.to_csv('ml_valid.csv', index=False)

    print('##### training set ####')
    for column in df_scores_train.columns:
        print(f'{column} | {df_scores_train.mean()[column]} - {df_scores_train.std()[column]}')

    print('\n##### validation set ####')
    for column in df_scores_valid.columns:
        print(f'{column} | {df_scores_valid.mean()[column]} - {df_scores_valid.std()[column]}')












