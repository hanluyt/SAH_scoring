# A Prognostic Scoring System for aSAH
Introduction
------------
Endovascular treatment for **aneurysmal subarachnoid hemorrhage** (aSAH) had improved significantly, but the existing prognostic scoring systems remained unchanged. In this study, 
we attempted to establish the multivariate models for the predictions of prognosis in patients after aSAH treated with endovascular approach,  **both in the general patient population 
and in the patients with good-grade aSAH on admission** (i.e. WENS≤3).

### Model selection
To determine the best model for the subsequent prognosis prediction, we compared four machine learning models, including the regularized logistic regression (RLR), linear support
vector machine (SVM), and random forest (RF), and a novel deep reinforcement learning algorithm, namely the ensemble imbalance learning framework with meta-sampler (MESA).
In this study, RLR, SVM and RF adopt the way of bagging ensemble learning, MESA adaptively resamples the training set by a reinforcement learning algorithm in iterations to 
get multiple classifiers and forms a cascade ensemble model.The model with the best performance was selected to evaluate the contributions of the clinically relevant features
to distinguishing the poor outcomes from the favorable ones.

### Prognosis Analysis
To test the superiority of the selected model in prediction capability, the classic models were built using Hunt-Hess, modified Fisher, and WFNS grades respectively. For these models, 
the sensitivity, specificity and AUC were calculated and compared using the independent test set. 

Data
--------------
This folder includes four csv files where `sah_train.csv` / `sah_test.csv` can be used to build prediction model for all patients after aSAH and `wfns0_train.csv` / `wnfs0_test.csv`
for patients with good-grade aSAH.

Usage
-------
### Running [main.py](https://github.com/hanluyt/SAH_scoring/blob/master/main.py) for model selection
* For all patients
```
python main.py --dataset sah
```
* For patients with good-grade aSAH
```
python main.py --dataset wfns0
```
### Running [prognosis.py](https://github.com/hanluyt/SAH_scoring/blob/master/prognosis.py) for prognosis analysis
* For all patients
```
python prognosis.py --dataset sah
```
* For patients with good-grade aSAH
```
python prognosis.py --dataset wfns0
```
Reference
--------
[MESA algorithm](https://github.com/ZhiningLiu1998/mesa)
      
