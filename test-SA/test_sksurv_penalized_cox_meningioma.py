import os.path
from fancyimpute import KNN, SoftImpute, IterativeImputer, NuclearNormMinimization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer

imp = KNN(k=5)
# imp2 =
# imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imp2 = SimpleImputer(missing_values='', strategy='most_frequent')
# imp3 = SimpleImputer(missing_values=None, strategy='most_frequent')
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis, \
    GradientBoostingSurvivalAnalysis

from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from time import time, strftime, localtime
import os
from pathlib import Path

time_str = strftime("%Y-%m-%d_%H-%M-%S", localtime())
dir_parent = '../notebooks/results'
directory = dir_parent + '/' + time_str
Path(directory).mkdir(parents=True, exist_ok=True)

alpha_minimal = 0.0000001
maximum_iterations = 1000000


def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots()
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min, coef, name + "   ",
            horizontalalignment="right",
            verticalalignment="center"
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")


from pathlib import Path

data_dir = '../csv/Meningioma/'

df_var1_demographics = pd.read_csv(Path(data_dir + 'meningioma.variable.time001.demographics.csv'), sep=',',
                                   encoding='utf-8')
df_var2_radiographics = pd.read_csv(Path(data_dir + 'meningioma.variable.time002.radiographics.csv'), sep=',',
                                    encoding='utf-8')
df_var2_radiomics = pd.read_csv(Path(data_dir + 'meningioma.variable.time002.radiomics.csv'), sep=',', encoding='utf-8')
df_var3_therapy = pd.read_csv(Path(data_dir + 'meningioma.variable.time003.therapy.csv'), sep=',', encoding='utf-8')
df_var4_pathology = pd.read_csv(Path(data_dir + 'meningioma.variable.time004.pathology.csv'), sep=',', encoding='utf-8')
df_outcome_1 = pd.read_csv(Path(data_dir + 'meningioma.outcome.time001.presentation.csv'), sep=',', encoding='utf-8')
df_outcome_2 = pd.read_csv(Path(data_dir + 'meningioma.outcome.time002.baseline_imaging.csv'), sep=',',
                           encoding='utf-8')
df_outcome_3 = pd.read_csv(Path(data_dir + 'meningioma.outcome.time003.therapy.csv'), sep=',', encoding='utf-8')
df_outcome_4 = pd.read_csv(Path(data_dir + 'meningioma.outcome.time004.pathology.csv'), sep=',', encoding='utf-8')
# %%
df_var_list = [df_var1_demographics, df_var2_radiographics, df_var2_radiomics, df_var3_therapy, df_var4_pathology]
df_outcome_list = [df_outcome_1, df_outcome_2, df_outcome_2, df_outcome_3, df_outcome_4]
dfs = {}
c_vars = ['A', 'B', 'C', 'D', 'E']
c_time = ['1', '2', '2', '3', '4']

for ii in range(len(df_var_list)):
    index = 4 - ii
    name = c_vars[index] + c_time[index]
    dfs[name] = df_var_list[index].merge(df_outcome_list[index], how='inner', on='ID')
    temp = name
    for other_var in range(index):
        # new_index = other_var
        name = c_vars[index - other_var - 1] + name
        dfs[name] = dfs[temp].merge(df_var_list[index - other_var - 1], how='inner', on='ID')
        temp = name

dfs_ = list(dfs.keys())
prog = len(dfs_)
targets_list = ['ID', 'LocalR.binary', 'LocalR.eventFreeTime', 'Death.binary', 'Death.eventFreeTime']
results = pd.DataFrame(columns=['name',
                                'LassoCoxnet_CINDEX',
                                'Elesticnet_CINDEX',
                                'Elasticnet_CINDEX_standardscaler',
                                'Elasticnet_CINDEX_gridsearch',
                                'CoxnetPred_CINDEX',
                                'RandomSurvivalForest_CINDEX',
                                ])
for df_index in range(len(dfs_)):
    dfs[dfs_[df_index]] = dfs[dfs_[df_index]][dfs[dfs_[df_index]]['Death.binary'].notna()]
    dfs[dfs_[df_index]] = dfs[dfs_[df_index]][dfs[dfs_[df_index]]['Death.eventFreeTime'].notna()]
    df_temp = dfs[dfs_[df_index]].fillna(value=None, method='ffill')

    X = df_temp.copy()
    y = Surv.from_dataframe(event='Death.binary',
                            time='Death.eventFreeTime',
                            data=df_temp[['Death.binary', 'Death.eventFreeTime']])
    # X = df_temp.drop(['Death.binary', 'Death.eventFreeTime'], axis=1)
    if df_index >= 1 and 'Grade' in X.columns:
        X = X.drop(columns=['Grade'])
    if df_index >= 1 and 'Grade.binary' in X.columns:
        X = X.drop(columns=['Grade.binary'])
    X = X.drop(columns=targets_list)
    Xt = X.copy()
    print(str(df_index + 1) + '/' + str(prog))
    print(dfs_[df_index])
    if len(Xt.columns) > 1:
        Xt = encode_categorical(Xt)
        Xt = OneHotEncoder().fit_transform(Xt)
        Xt = pd.DataFrame(imp.fit_transform(Xt), columns=Xt.columns)
        # Xt = imp.transform(Xt)

    print(Xt.round(2).head())
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=0)

    print("================================================================================================")
    print("Ridge -- CoxPHSurvivalAnalysis")
    print("================================================================================================")

    #
    # alphas = 10. ** np.linspace(-4, 4, 70)
    # coefficients = {}
    #
    # cph = CoxPHSurvivalAnalysis()
    # for alpha in alphas:
    #     cph.set_params(alpha=alpha)
    #     cph.fit(X_train, y_train)
    #     key = round(alpha, 5)
    #     coefficients[key] = cph.coef_
    #
    # coefficients = (pd.DataFrame
    #                 .from_dict(coefficients)
    #                 .rename_axis(index="feature", columns="alpha")
    #                 .set_index(Xt.columns))
    #
    # plot_coefficients(coefficients, n_highlight=5)
    # plt.show()
    # plt.savefig(directory+'/'+str(dfs_[df_index])+'_coefficients_ridge.png')

    print("================================================================================================")
    print("Lasso -- CoxNetSurvivalAnalysis")
    print("================================================================================================")

    cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.0001)#, alpha_min_ratio=alpha_minimal, max_iter=maximum_iterations)
    cox_lasso.fit(X_train, y_train)

    coefficients_lasso = pd.DataFrame(
        cox_lasso.coef_,
        index=Xt.columns,
        columns=np.round(cox_lasso.alphas_, 5)
    )

    plot_coefficients(coefficients_lasso, n_highlight=3)
    plt.show()
    plt.savefig(directory + '/' + str(dfs_[df_index]) + '_coefficients_lasso.png')

    cindex1 = cox_lasso.score(X_test, y_test)
    print(round(cindex1, 3))
    # results = results.append({'name': dfs_[df_index], 'LassoCoxnet_CINDEX': cindex}, ignore_index=True)
    print("================================================================================================")
    print("ElasticNet -- CoxNetSurvivalAnalysis")
    print("================================================================================================")

    cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=alpha_minimal, max_iter=maximum_iterations)
    cox_elastic_net.fit(X_train, y_train)

    coefficients_elastic_net = pd.DataFrame(
        cox_elastic_net.coef_,
        index=Xt.columns,
        columns=np.round(cox_elastic_net.alphas_, 5)
    )

    plot_coefficients(coefficients_elastic_net, n_highlight=5)
    plt.show()
    plt.savefig(directory + '/' + str(dfs_[df_index]) + '_coefficients_elastic_net.png')

    cindex2 = cox_elastic_net.score(X_test, y_test)
    print(round(cindex2, 3))
    # results = results.append({'name': dfs_[df_index], 'Elesticnet_CINDEX': cindex}, ignore_index=True)

    print("================================================================================================")
    print("Choose penalty strength -- Alpha")
    print("================================================================================================")

    import warnings
    from sklearn.exceptions import FitFailedWarning
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    print("Test 1")
    coxnet_pipe = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=alpha_minimal*0.001, max_iter=maximum_iterations*1000)
    )
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(X_train, y_train)

    cindex3 = coxnet_pipe.score(X_test, y_test)
    print(round(cindex3, 3))
    # results = results.append({'name': dfs_[df_index], 'Elasticnet_CINDEX_standardscaler': cindex}, ignore_index=True)

    print("Test 1")
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(),
                      CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=alpha_minimal*0.001, max_iter=maximum_iterations*1000)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=4).fit(X_train, y_train)

    cindex4 = gcv.score(X_test, y_test)
    print(round(cindex4, 3))
    # results = results.append({'name': dfs_[df_index], 'Elasticnet_CINDEX_gridsearch': cindex}, ignore_index=True)

    cv_results = pd.DataFrame(gcv.cv_results_)

    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score

    fig, ax = plt.subplots()
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.show()

    plt.savefig(directory + '/' + str(dfs_[df_index]) + '_alpha_ridge.png')

    print("Test 2")
    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    best_coefs = pd.DataFrame(
        best_model.coef_,
        index=Xt.columns,
        columns=["coefficient"]
    )

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print("Number of non-zero coefficients: {}".format(non_zero))

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots()
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.show()

    plt.savefig(directory + '/' + str(dfs_[df_index]) + 'non_zero_coefs.png')

    print("================================================================================================")
    print("Survival and Cumulative hazard function")
    print("================================================================================================")

    coxnet_pred = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio=0.9,
                               alpha_min_ratio=alpha_minimal*0.001,
                               max_iter=maximum_iterations*1000,
                               fit_baseline_model=True)
    )
    coxnet_pred.set_params(**gcv.best_params_)
    coxnet_pred.fit(X_train, y_train)

    y_pred = coxnet_pred.predict(X_test)

    surv_fns = coxnet_pred.predict_survival_function(X_test)
    group_prediction = ['High' if pred > np.median(y_pred) else 'Low' for pred in y_pred]

    cindex5 = coxnet_pred.score(X_test, y_test)
    print(round(cindex5, 3))
    # results = results.append({'name': dfs_[df_index], 'CoxnetPred_CINDEX': cindex}, ignore_index=True)

    time_points = np.quantile(y["Death.eventFreeTime"], np.linspace(0, 0.90, 100))
    legend_handles = []
    legend_labels = []
    _, ax = plt.subplots()

    import matplotlib.lines as mlines

    line_high = mlines.Line2D([], [], color="C1", label="High")
    line_low = mlines.Line2D([], [], color="C0", label="Low")
    legend_handles.append(line_high)
    legend_handles.append(line_low)

    for fn, label in zip(surv_fns, group_prediction):
        if label == 'High':
            color = 'C1'
        else:
            color = 'C0'
        line, = ax.step(time_points, fn(time_points), where="post",
                        color=color, alpha=0.5)
    ax.legend(handles=legend_handles)  # , legend_labels)
    ax.set_xlabel("time")
    ax.set_ylabel("Survival probability")
    ax.grid(True)
    plt.show()

    plt.savefig(directory + '/' + str(dfs_[df_index]) + 'coxnet_survival_function.png')

    print("================================================================================================")
    print("Survival and Cumulative hazard function")
    print("================================================================================================")

    rsf_pred = make_pipeline(
        StandardScaler(),
        RandomSurvivalForest(n_estimators=1000,
                             min_samples_split=10,
                             min_samples_leaf=15,
                             max_features="sqrt",
                             n_jobs=-1,
                             random_state=0)
    )
    rsf_pred.fit(X_train, y_train)

    y_pred = rsf_pred.predict(X_test)

    cindex6 = rsf_pred.score(X_test, y_test)
    print(round(cindex6, 3))
    # results = results.append({'name': dfs_[df_index], 'RandomSurvivalForest_CINDEX': cindex}, ignore_index=True)

    surv_fns = rsf_pred.predict_survival_function(X_test)
    group_prediction = ['High' if pred > np.median(y_pred) else 'Low' for pred in y_pred]

    time_points = np.quantile(y["Death.eventFreeTime"], np.linspace(0, 0.90, 100))
    _, ax = plt.subplots()

    for fn, label in zip(surv_fns, group_prediction):
        if label == 'High':
            color = 'C1'
        else:
            color = 'C0'
        line, = ax.step(time_points, fn(time_points), where="post",
                        color=color, alpha=0.5)
    ax.legend(handles=legend_handles)  # , legend_labels)
    ax.set_xlabel("time")
    ax.set_ylabel("Survival probability")
    ax.grid(True)
    plt.show()

    plt.savefig(directory + '/' + str(dfs_[df_index]) + 'rsf_survival_function.png')

    print("================================================================================================")
    print('Appending results to dataframe')
    print("================================================================================================")
    results = results.append({'name': dfs_[df_index],
                              'LassoCoxnet_CINDEX': cindex1,
                              'Elesticnet_CINDEX': cindex2,
                              'Elasticnet_CINDEX_standardscaler': cindex3,
                              'Elasticnet_CINDEX_gridsearch': cindex4,
                              'CoxnetPred_CINDEX': cindex5,
                              'RandomSurvivalForest_CINDEX': cindex6}, ignore_index=True)

results.to_csv(directory + '/results.csv')
# if __name__ == "__main__":
#
