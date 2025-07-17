import pandas as pd
import csv
import os
import numpy as np
from SurvivalAnalysis.pycaret_wrapper.oop import SurvivalExperiment
from SurvivalAnalysis.pycaret_wrapper.oop import *
from pycaret.regression.oop import RegressionExperiment
import lifelines
exp1 = SurvivalExperiment()
data_dir = '..\\csv\\'
df = pd.read_csv("../csv/SA/NKI_cleaned.csv", sep=',', encoding='utf-8')

features_to_drop = df.columns[16:]
df = df.drop(features_to_drop, axis=1)
for cols in df.columns:
    print(cols)

# from copy import deepcopy
# from pathlib import Path
# data_dir = 'A:\\MEDomics\\WORKSPACE_Summer2022_meningioma-20220523T192422Z-001\\WORKSPACE_Summer2022_meningioma\\data\\csv\\'
#
# # load csv to dataframe
# df_var1_demographics = pd.read_csv(Path(data_dir+'meningioma.variable.time001.demographics.csv'), sep=',', encoding='utf-8')
# df_var2_radiographics = pd.read_csv(Path(data_dir+'meningioma.variable.time002.radiographics.csv'), sep=',', encoding='utf-8')
# df_var2_radiomics = pd.read_csv(Path(data_dir+'meningioma.variable.time002.radiomics.csv'), sep=',', encoding='utf-8')
# # df_var3_therapy = pd.read_csv(Path(data_dir+'meningioma.variable.time003.therapy.csv'), sep=',', encoding='utf-8')
# # df_var4_pathology = pd.read_csv(Path(data_dir+'meningioma.variable.time004.pathology.csv'), sep=',', encoding='utf-8')
# df_outcome_1 = pd.read_csv(Path(data_dir + 'meningioma.outcome.time001.presentation.csv'), sep=',', encoding='utf-8')
# df_outcome_2 = pd.read_csv(Path(data_dir + 'meningioma.outcome.time002.baseline_imaging.csv'), sep=',', encoding='utf-8')
# # df_outcome_3 = pd.read_csv(Path(data_dir+'meningioma.outcome.time003.therapy.csv'), sep=',', encoding='utf-8')
# # df_outcome_4 = pd.read_csv(Path(data_dir+'meningioma.outcome.time004.pathology.csv'), sep=',', encoding='utf-8')
#
# dff = df_var1_demographics.merge(df_outcome_2, how='inner', on='ID')
# dff = dff.merge(df_var2_radiographics, how='inner', on='ID')
# # dff = dff.merge(df_var2_radiomics, how='inner', on='ID')
#
# dfff = deepcopy(dff[dff['Death.binary'].notna()])
#
# dff = dff[dff['Death.binary'].notna()]
# dff = dff[dff['Death.eventFreeTime'].notna()]
# dff = dff.fillna(value=None, method='ffill')
#
# exp1.setup(data=dff[dff['Death.binary'].notna()],
#            target=['Death.binary', 'Death.eventFreeTime'],
#            ignore_features=['ID','LocalR.binary', 'LocalR.eventFreeTime'],
#            keep_features=['Death.eventFreeTime'],
#            # data_split_shuffle=True,
#            # data_split_stratify=True,
#            fold_strategy='kfold',
#            log_data=True,
#            # fold=2,
#            numeric_imputation='median',
#            categorical_imputation='mode',
#            imputation_type='simple',
#
#            # transform_target=False,
#            # transformation=True,
#            preprocess=True,
#            normalize=False,
#            # feature_selection=True,
#            # feature_selection_method='sequential',
#            remove_outliers=True,
#            remove_multicollinearity=True,
#            n_jobs=-1,
#
#            verbose=True)

exp1.setup(data=df,
           target=['eventdeath', 'timerecurrence'],
           ignore_features=['Patient','ID', 'barcode'],
           keep_features=['timerecurrence'],
           fold=10,
           preprocess=True,
           normalize=False,
           n_jobs=1,
           verbose=False)



from sksurv.metrics import *
import sksurv.metrics as s_metrics
from sklearn.metrics import *
exp1.add_metric('logloss', 'Log Loss', log_loss, greater_is_better = False)

# exp1.add_metric('ci', 'C-Index', s_metrics.concordance_index_censored, greater_is_better = False)

def ci_score(y_true, y_pred):
    return s_metrics.concordance_index_censored(y_true, y_pred)

# exp1.add_metric()
print("===========================================================")
print("available models:")
print("===========================================================")
print(exp1.models())
print(exp1.get_metrics())

print("===========================================================")
print("compare models:")
print("===========================================================")
exp1.compare_models(sort='ci', errors='raise')
print("===========================================================")
print("training model:")
print("===========================================================")
model = exp1.create_model('coxnet',  cross_validation=True, return_train_score=True)
# best_model = exp1.compare_models()
print("===========================================================")
print("prediction:")
print("===========================================================")
exp1.predict_model(model)
print('===========================================================')
print('===========================================================')


print("===========================================================")
print("available models:")
print("===========================================================")
print(exp1.models())
print(exp1.get_metrics())

print("===========================================================")
print("compare models:")
print("===========================================================")
models_ = exp1.compare_models(sort='ci-ipcw', errors='raise', n_select=3)
print("===========================================================")
print("training model:")
print("===========================================================")
model1 = exp1.create_model('coxnet', cross_validation=True, return_train_score=True)



print("===========================================================")
print("Plotting:")
print("===========================================================")

exp1.plot_model(model1, plot='plot_coefficients')
exp1.plot_model(model1, plot='plot_cindex')
exp1.plot_model(model1, 'plot_survival_curve')
exp1.plot_model(model1, plot='plot_cauc')
exp1.plot_model(model1, plot='plot_nzcoefs')
exp1.plot_model(model1, 'plot_coefficients')

print('===========================================================')
print('===========================================================')
print('FIN')
