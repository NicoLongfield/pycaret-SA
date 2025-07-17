import datetime
import os
import logging
import time
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gc
import traceback
import numpy as np  # type: ignore
import pandas as pd
from joblib.memory import Memory
from numpy import ndarray
from pandas import Series
from sklearn.base import clone  # type: ignore
from copy import deepcopy
from unittest.mock import patch
import plotly.express as px  # type: ignore
import scikitplot as skplt  # type: ignore
from IPython.display import display as ipython_display
from joblib.memory import Memory
from packaging import version
from pandas.io.formats.style import Styler

# sys.path.insert(1, '../')
# os.chdir('A:\\MEDomics\\MED\\MEDml\\pycaret_local')

from pycaret.internal.plots.helper import MatplotlibDefaultDPI
from pycaret.internal.plots.yellowbrick import show_yellowbrick_plot
# from
from pycaret.internal.meta_estimators import (
    CustomProbabilityThresholdClassifier,
    PowerTransformedTargetRegressor,
    get_estimator_from_meta_estimator,
)
from pycaret.utils._dependencies import _check_soft_dependencies

from pycaret.utils.generic import (
    can_early_stop,
    color_df,
    get_label_encoder,
    id_or_display_name,
    nullcontext,
    true_warm_start,
)

import pycaret.containers.metrics.regression
import pycaret.containers.models.regression
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
from pycaret.internal.validation import is_fitted, is_sklearn_cv_generator

from pycaret.internal.logging import redirect_output
from pycaret.internal.display import CommonDisplay
from pycaret.internal.logging import get_logger
from pycaret.internal.parallel.parallel_backend import ParallelBackend
from pycaret.utils.generic import get_ml_task
# Own module
from pycaret.survival_analysis.pycaret_wrapper.pipeline import Pipeline as InternalPipeline
from pycaret.survival_analysis.pycaret_wrapper.pipeline import (estimator_pipeline,
                                                                   get_pipeline_estimator_label,
                                                                   get_pipeline_fit_kwargs)
from pycaret.survival_analysis.preprocessor import Preprocessor
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.utils.generic import MLUsecase, highlight_setup
from pycaret.utils.constants import DATAFRAME_LIKE, TARGET_LIKE
from pycaret.loggers.base_logger import BaseLogger

import pycaret.survival_analysis.pycaret_wrapper.models as sa_models
import pycaret.survival_analysis.pycaret_wrapper.metrics as sa_metrics
from pycaret.survival_analysis.pycaret_wrapper.models import *
import sksurv.datasets as sa_data
from sksurv.util import Surv

import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.util import Surv
from sksurv.nonparametric import *
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score, brier_score
LOGGER = get_logger()


class SurvivalExperiment(_SupervisedExperiment, Preprocessor):

    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.SURVIVAL_ANALYSIS
        self.exp_name_log = "SA-default-name"
        self._variable_keys = self._variable_keys.union(
            {
                "transform_target_param",
                "transform_target_method_param",
                "target_event_param",
                "survival_train",
                "time_range",
            }
        )
        self._available_plots = {
            "plot_grouped_survival": "Grouped Survival curves",
            "plot_survival_curve": "Survival curve",
            "plot_nzcoefs" : "Non-zero coefficients",
            "plot_cauc": "Cumulative AUC",
            "plot_cindex" : "C-index",
            "plot_coefficients": "Survival Coefficients",
            "auc" : "AUC",
            "boundary": "Boundary",
            "calibration": "Calibration",
            "confusion_matrix": "Confusion Matrix",
            "dimension": "Dimension",
            "distance": "Distance",
            "feature": "Feature",
            "feature_all": "Feature All",
            "gain": "Gain",
            "elbow": "Elbow",
            "pr": "PR",
            "silhouette": "Silhouette",
            "ks": "KS",
            "residuals_interactive": "Residuals Interactive",
            "pipeline": "Pipeline Plot",
            "parameter": "Hyperparameters",
            "residuals": "Residuals",
            "error": "Prediction Error",
            "cooks": "Cooks Distance",
            "learning": "Learning Curve",
            "manifold": "Manifold Learning",
            "vc": "Validation Curve",
            "test_plot": "test_plot",
            "coefficients_surv": "coefficients_surv",
            "partial_effects_on_outcome": "partial_effects_on_outcome",
        }

    @property
    def X_train(self):
        """Feature set of the training set."""
        # For survival analysis, we need to keep the time column in the features
        X_train = self.train.drop([self.target_param], axis=1, errors='ignore')
        return X_train

    @property
    def X_test(self):
        """Feature set of the test set."""
        # For survival analysis, we need to keep the time column in the features
        X_test = self.test.drop([self.target_param], axis=1, errors='ignore')
        return X_test

    @property
    def test(self):
        """Test set."""
        return self.dataset.loc[self.idx[1], :]

    @property
    def test_transformed(self):
        """Transformed test set."""
        return pd.concat([self.X_test_transformed, self.y_test_transformed], axis=1)

    @property
    def X_transformed(self):
        """Transformed feature set."""
        return pd.concat([self.X_train_transformed, self.X_test_transformed])

    @property
    def y_transformed(self):
        """Transformed target column."""
        return pd.concat([self.y_train_transformed, self.y_test_transformed])

    @property
    def X_train_transformed(self):
        """Transformed feature set of the training set."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            return self.pipeline.transform(X=self.X_train, y=self.y_train, filter_train_only=False)[0]
        return None

    @property
    def X_test_transformed(self):
        """Transformed feature set of the test set."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            return self.pipeline.transform(self.X_test, self.y_test)[0]
        return None

    @property
    def y_train_transformed(self):
        """Transformed target column of the training set."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            return self.pipeline.transform(X=self.X_train, y=self.y_train, filter_train_only=False)[1]
        return None

    @property
    def y_test_transformed(self):
        """Transformed target column of the test set."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            return self.pipeline.transform(self.X_test, self.y_test)[1]
        return None

    @property
    def dataset_transformed(self):
        """Transformed dataset."""
        return pd.concat([self.train_transformed, self.test_transformed])

    @property
    def train_transformed(self):
        """Transformed training set."""
        return pd.concat([self.X_train_transformed, self.y_train_transformed], axis=1)

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in sa_models.get_all_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = (
            sa_models.get_all_model_containers(
                self, raise_errors=raise_errors
            )
        )
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return sa_metrics.get_all_metric_containers(
            self.variables, raise_errors=raise_errors
        )

    def _get_default_plots_to_log(self) -> List[str]:
        return ["residuals", "error", "feature"]

    def _calculate_metrics(
        self,
        y_test,
        y_train,
        X_test,
        X_train,
        pred,
        pred_prob,
        weights: Optional[list] = None,
        **additional_kwargs,
    ) -> dict:
        """
        Calculate all metrics in _all_metrics.
        """
        from pycaret.survival_analysis.pycaret_wrapper.utils import calculate_metrics

        with redirect_output(self.logger):
            try:
                return calculate_metrics(
                    metrics=self._all_metrics,
                    y_test=y_test,
                    y_train=y_train,
                    X_test=X_test,
                    X_train=X_train,
                    pred=pred,
                    pred_proba=pred_prob,
                    weights=weights,
                    **additional_kwargs,
                )
            except Exception:
                ml_usecase = get_ml_task(y_test)
                if ml_usecase == MLUsecase.CLASSIFICATION:
                    metrics = pycaret.containers.metrics.classification.get_all_metric_containers(
                        self.variables, True
                    )
                elif ml_usecase == MLUsecase.REGRESSION:
                    metrics = (
                        pycaret.containers.metrics.regression.get_all_metric_containers(
                            self.variables, True
                        )
                    )
                return calculate_metrics(
                    metrics=metrics,  # type: ignore
                    y_test=y_test,
                    y_train=y_train,
                    X_test=X_test,
                    X_train=X_train,
                    pred=pred,
                    pred_proba=pred_prob,
                    weights=weights,
                    **additional_kwargs,
                )

    def setup(
            self,
            data: Optional[DATAFRAME_LIKE] = None,
            data_func: Optional[Callable[[], DATAFRAME_LIKE]] = None,
            target: List[str] = None,
            time_range_percentiles: List[float] = [10, 50],
            train_size: float = 0.7,
            test_data: Optional[DATAFRAME_LIKE] = None,
            ordinal_features: Optional[Dict[str, list]] = None,
            numeric_features: Optional[List[str]] = None,
            categorical_features: Optional[List[str]] = None,
            date_features: Optional[List[str]] = None,
            text_features: Optional[List[str]] = None,
            ignore_features: Optional[List[str]] = None,
            keep_features: Optional[List[str]] = None,
            preprocess: bool = True,
            imputation_type: Optional[str] = "simple",
            numeric_imputation: str = "mean",
            categorical_imputation: str = "constant",
            iterative_imputation_iters: int = 5,
            numeric_iterative_imputer: Union[str, Any] = "lightgbm",
            categorical_iterative_imputer: Union[str, Any] = "lightgbm",
            text_features_method: str = "tf-idf",
            max_encoding_ohe: int = 5,
            encoding_method: Optional[Any] = None,
            polynomial_features: bool = False,
            polynomial_degree: int = 2,
            low_variance_threshold: float = 0,
            remove_multicollinearity: bool = False,
            multicollinearity_threshold: float = 0.9,
            bin_numeric_features: Optional[List[str]] = None,
            remove_outliers: bool = False,
            outliers_method: str = "iforest",
            outliers_threshold: float = 0.05,
            transformation: bool = False,
            transformation_method: str = "yeo-johnson",
            normalize: bool = False,
            normalize_method: str = "zscore",
            pca: bool = False,
            pca_method: str = "linear",
            pca_components: Union[int, float] = 1.0,
            feature_selection: bool = False,
            feature_selection_method: str = "classic",
            feature_selection_estimator: Union[str, Any] = "lightgbm",
            n_features_to_select: int = 10,
            transform_target: bool = False,
            transform_target_method: str = "yeo-johnson",
            custom_pipeline: Optional[Any] = None,
            data_split_shuffle: bool = True,
            data_split_stratify: Union[bool, List[str]] = False,
            fold_strategy: Union[str, Any] = "kfold",
            fold: int = 10,
            fold_shuffle: bool = False,
            fold_groups: Optional[Union[str, pd.DataFrame]] = None,
            n_jobs: Optional[int] = -1,
            use_gpu: bool = False,
            html: bool = True,
            session_id: Optional[int] = None,
            system_log: Union[bool, str, logging.Logger] = True,
            log_experiment: Union[
                bool, str, BaseLogger, List[Union[str, BaseLogger]]
            ] = False,
            experiment_name: Optional[str] = None,
            experiment_custom_tags: Optional[Dict[str, Any]] = None,
            log_plots: Union[bool, list] = False,
            log_profile: bool = False,
            log_data: bool = False,
            verbose: bool = True,
            memory: Union[bool, str, Memory] = True,
            profile: bool = False,
            profile_kwargs: Dict[str, Any] = None,
    ):

        """
        This function initializes the training environment and creates the transformation
        pipeline. Setup function must be called before executing any other function. It takes
        two mandatory parameters: ``data`` and ``target``. All the other parameters are
        optional.

        Example
        -------
        >>> from pycaret_local.datasets import get_data
        >>> juice = get_data('juice')
        >>> from pycaret_local.classification import *
        >>> exp_name = setup(data = juice,  target = 'Purchase')


        data: dataframe-like = None
            Data set with shape (n_samples, n_features), where n_samples is the
            number of samples and n_features is the number of features. If data
            is not a pandas dataframe, it's converted to one using default column
            names.


        data_func: Callable[[], DATAFRAME_LIKE] = None
            The function that generate ``data`` (the dataframe-like input). This
            is useful when the dataset is large, and you need parallel operations
            such as ``compare_models``. It can avoid boradcasting large dataset
            from driver to workers. Notice one and only one of ``data`` and
            ``data_func`` must be set.


        target: int, str or sequence, default = -1
            If int or str, respectivcely index or name of the target column in data.
            The default value selects the last column in the dataset. If sequence,
            it should have shape (n_samples,). The target can be either binary or
            multiclass.


        train_size: float, default = 0.7
            Proportion of the dataset to be used for training and validation. Should be
            between 0.0 and 1.0.


        test_data: dataframe-like or None, default = None
            If not None, test_data is used as a hold-out set and `train_size` parameter
            is ignored. The columns of data and test_data must match. If it's a pandas
            dataframe, the indices must match as well.


        ordinal_features: dict, default = None
            Categorical features to be encoded ordinally. For example, a categorical
            feature with 'low', 'medium', 'high' values where low < medium < high can
            be passed as ordinal_features = {'column_name' : ['low', 'medium', 'high']}.


        numeric_features: list of str, default = None
            If the inferred data types are not correct, the numeric_features param can
            be used to define the data types. It takes a list of strings with column
            names that are numeric.


        categorical_features: list of str, default = None
            If the inferred data types are not correct, the categorical_features param
            can be used to define the data types. It takes a list of strings with column
            names that are categorical.


        date_features: list of str, default = None
            If the inferred data types are not correct, the date_features param can be
            used to overwrite the data types. It takes a list of strings with column
            names that are DateTime.


        text_features: list of str, default = None
            Column names that contain a text corpus. If None, no text features are
            selected.


        ignore_features: list of str, default = None
            ignore_features param can be used to ignore features during preprocessing
            and model training. It takes a list of strings with column names that are
            to be ignored.


        keep_features: list of str, default = None
            keep_features param can be used to always keep specific features during
            preprocessing, i.e. these features are never dropped by any kind of
            feature selection. It takes a list of strings with column names that are
            to be kept.


        preprocess: bool, default = True
            When set to False, no transformations are applied except for train_test_split
            and custom transformations passed in ``custom_pipeline`` param. Data must be
            ready for modeling (no missing values, no dates, categorical data encoding),
            when preprocess is set to False.


        imputation_type: str or None, default = 'simple'
            The type of imputation to use. Can be either 'simple' or 'iterative'.
            If None, no imputation of missing values is performed.


        numeric_imputation: int, float or str, default = 'mean'
            Imputing strategy for numerical columns. Ignored when ``imputation_type=
            iterative``. Choose from:
                - "drop": Drop rows containing missing values.
                - "mean": Impute with mean of column.
                - "median": Impute with median of column.
                - "mode": Impute with most frequent value.
                - "knn": Impute using a K-Nearest Neighbors approach.
                - int or float: Impute with provided numerical value.


        categorical_imputation: str, default = 'mode'
            Imputing strategy for categorical columns. Ignored when ``imputation_type=
            iterative``. Choose from:
                - "drop": Drop rows containing missing values.
                - "mode": Impute with most frequent value.
                - str: Impute with provided string.


        iterative_imputation_iters: int, default = 5
            Number of iterations. Ignored when ``imputation_type=simple``.


        numeric_iterative_imputer: str or sklearn estimator, default = 'lightgbm'
            Regressor for iterative imputation of missing values in numeric features.
            If None, it uses LGBClassifier. Ignored when ``imputation_type=simple``.


        categorical_iterative_imputer: str or sklearn estimator, default = 'lightgbm'
            Regressor for iterative imputation of missing values in categorical features.
            If None, it uses LGBClassifier. Ignored when ``imputation_type=simple``.


        text_features_method: str, default = "tf-idf"
            Method with which to embed the text features in the dataset. Choose
            between "bow" (Bag of Words - CountVectorizer) or "tf-idf" (TfidfVectorizer).
            Be aware that the sparse matrix output of the transformer is converted
            internally to its full array. This can cause memory issues for large
            text embeddings.


        max_encoding_ohe: int, default = 5
            Categorical columns with `max_encoding_ohe` or less unique values are
            encoded using OneHotEncoding. If more, the `encoding_method` estimator
            is used. Note that columns with exactly two classes are always encoded
            ordinally. Set to below 0 to always use OneHotEncoding.


        encoding_method: category-encoders estimator, default = None
            A `category-encoders` estimator to encode the categorical columns
            with more than `max_encoding_ohe` unique values. If None,
            `category_encoders.leave_one_out.LeaveOneOutEncoder` is used.


        polynomial_features: bool, default = False
            When set to True, new features are derived using existing numeric features.


        polynomial_degree: int, default = 2
            Degree of polynomial features. For example, if an input sample is two dimensional
            and of the form [a, b], the polynomial features with degree = 2 are:
            [1, a, b, a^2, ab, b^2]. Ignored when ``polynomial_features`` is not True.


        low_variance_threshold: float or None, default = 0
            Remove features with a training-set variance lower than the provided
            threshold. The default is to keep all features with non-zero variance,
            i.e. remove the features that have the same value in all samples. If
            None, skip this treansformation step.


        remove_multicollinearity: bool, default = False
            When set to True, features with the inter-correlations higher than the defined
            threshold are removed. When two features are highly correlated with each other,
            the feature that is less correlated with the target variable is removed. Only
            considers numeric features.


        multicollinearity_threshold: float, default = 0.9
            Threshold for correlated features. Ignored when ``remove_multicollinearity``
            is not True.


        bin_numeric_features: list of str, default = None
            To convert numeric features into categorical, bin_numeric_features parameter can
            be used. It takes a list of strings with column names to be discretized. It does
            so by using 'sturges' rule to determine the number of clusters and then apply
            KMeans algorithm. Original values of the feature are then replaced by the
            cluster label.


        remove_outliers: bool, default = False
            When set to True, outliers from the training data are removed using an
            Isolation Forest.


        outliers_method: str, default = "iforest"
            Method with which to remove outliers. Ignored when `remove_outliers=False`.
            Possible values are:
                - 'iforest': Uses sklearn's IsolationForest.
                - 'ee': Uses sklearn's EllipticEnvelope.
                - 'lof': Uses sklearn's LocalOutlierFactor.


        outliers_threshold: float, default = 0.05
            The percentage of outliers to be removed from the dataset. Ignored
            when ``remove_outliers=False``.


        transformation: bool, default = False
            When set to True, it applies the power transform to make data more Gaussian-like.
            Type of transformation is defined by the ``transformation_method`` parameter.


        transformation_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'.
            Ignored when ``transformation`` is not True.


        normalize: bool, default = False
            When set to True, it transforms the features by scaling them to a given
            range. Type of scaling is defined by the ``normalize_method`` parameter.


        normalize_method: str, default = 'zscore'
            Defines the method for scaling. By default, normalize method is set to 'zscore'
            The standard zscore is calculated as z = (x - u) / s. Ignored when ``normalize``
            is not True. The other options are:

            - minmax: scales and translates each feature individually such that it is in
            the range of 0 - 1.
            - maxabs: scales and translates each feature individually such that the
            maximal absolute value of each feature will be 1.0. It does not
            shift/center the data, and thus does not destroy any sparsity.
            - robust: scales and translates each feature according to the Interquartile
            range. When the dataset contains outliers, robust scaler often gives
            better results.


        pca: bool, default = False
            When set to True, dimensionality reduction is applied to project the data into
            a lower dimensional space using the method defined in ``pca_method`` parameter.


        pca_method: str, default = 'linear'
            Method with which to apply PCA. Possible values are:
                - 'linear': Uses Singular Value  Decomposition.
                - kernel: Dimensionality reduction through the use of RBF kernel.
                - incremental: Similar to 'linear', but more efficient for large datasets.


        pca_components: int or float, default = 1.0
            Number of components to keep. If >1, it selects that number of
            components. If <= 1, it selects that fraction of components from
            the original features. The value must be smaller than the number
            of original features. This parameter is ignored when `pca=False`.


        feature_selection: bool, default = False
            When set to True, a subset of features is selected based on a feature
            importance score determined by ``feature_selection_estimator``.


        feature_selection_method: str, default = 'classic'
            Algorithm for feature selection. Choose from:
                - 'univariate': Uses sklearn's SelectKBest.
                - 'classic': Uses sklearn's SelectFromModel.
                - 'sequential': Uses sklearn's SequtnailFeatureSelector.


        feature_selection_estimator: str or sklearn estimator, default = 'lightgbm'
            Classifier used to determine the feature importances. The
            estimator should have a `feature_importances_` or `coef_`
            attribute after fitting. If None, it uses LGBRegressor. This
            parameter is ignored when `feature_selection_method=univariate`.


        n_features_to_select: int, default = 10
            The number of features to select. Note that this parameter doesn't
            take features in ``ignore_features`` or ``keep_features`` into account
            when counting.


        transform_target: bool, default = False
            When set to True, target variable is transformed using the method defined in
            ``transform_target_method`` param. Target transformation is applied separately
            from feature transformations.


        transform_target_method: str, default = 'yeo-johnson'
            Defines the method for transformation. By default, the transformation method is
            set to 'yeo-johnson'. The other available option for transformation is 'quantile'.
            Ignored when ``transform_target`` is not True.

        custom_pipeline: list of (str, transformer), dict or Pipeline, default = None
            Addidiotnal custom transformers. If passed, they are applied to the
            pipeline last, after all the build-in transformers.


        data_split_shuffle: bool, default = True
            When set to False, prevents shuffling of rows during 'train_test_split'.


        data_split_stratify: bool or list, default = False
            Controls stratification during 'train_test_split'. When set to True, will
            stratify by target column. To stratify on any other columns, pass a list of
            column names. Ignored when ``data_split_shuffle`` is False.


        fold_strategy: str or sklearn CV generator object, default = 'kfold'
            Choice of cross validation strategy. Possible values are:

            * 'kfold'
            * 'groupkfold'
            * 'timeseries'
            * a custom CV generator object compatible with scikit-learn.


        fold: int, default = 10
            Number of folds to be used in cross validation. Must be at least 2. This is
            a global setting that can be over-written at function level by using ``fold``
            parameter. Ignored when ``fold_strategy`` is a custom object.


        fold_shuffle: bool, default = False
            Controls the shuffle parameter of CV. Only applicable when ``fold_strategy``
            is 'kfold' or 'stratifiedkfold'. Ignored when ``fold_strategy`` is a custom
            object.


        fold_groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when 'GroupKFold' is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in the training dataset. When string is passed, it is interpreted
            as the column name in the dataset containing group labels.


        n_jobs: int, default = -1
            The number of jobs to run in parallel (for functions that supports parallel
            processing) -1 means using all processors. To run all functions on single
            processor set n_jobs to None.


        use_gpu: bool or str, default = False
            When set to True, it will use GPU for training with algorithms that support it,
            and fall back to CPU if they are unavailable. When set to 'force', it will only
            use GPU-enabled algorithms and raise exceptions when they are unavailable. When
            False, all algorithms are trained using CPU only.

            GPU enabled algorithms:

            - Extreme Gradient Boosting, requires no further installation

            - CatBoost Classifier, requires no further installation
            (GPU is only enabled when data > 50,000 rows)

            - Light Gradient Boosting Machine, requires GPU installation
            https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html

            - Linear Regression, Lasso Regression, Ridge Regression, K Neighbors Regressor,
            Random Forest, Support Vector Regression, Elastic Net requires cuML >= 0.15
            https://github.com/rapidsai/cuml


        html: bool, default = True
            When set to False, prevents runtime display of monitor. This must be set to False
            when the environment does not support IPython. For example, command line terminal,
            Databricks Notebook, Spyder and other similar IDEs.


        session_id: int, default = None
            Controls the randomness of experiment. It is equivalent to 'random_state' in
            scikit-learn. When None, a pseudo random number is generated. This can be used
            for later reproducibility of the entire experiment.


        log_experiment: bool, default = False
            A (list of) PyCaret ``BaseLogger`` or str (one of 'mlflow', 'wandb')
            corresponding to a logger to determine which experiment loggers to use.
            Setting to True will use just MLFlow.
            If ``wandb`` (Weights & Biases) is installed, will also log there.


        system_log: bool or str or logging.Logger, default = True
            Whether to save the system logging file (as logs.log). If the input
            is a string, use that as the path to the logging file. If the input
            already is a logger object, use that one instead.


        experiment_name: str, default = None
            Name of the experiment for logging. Ignored when ``log_experiment`` is False.


        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified
            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


        log_plots: bool or list, default = False
            When set to True, certain plots are logged automatically in the ``MLFlow`` server.
            To change the type of plots to be logged, pass a list containing plot IDs. Refer
            to documentation of ``plot_model``. Ignored when ``log_experiment`` is False.


        log_profile: bool, default = False
            When set to True, data profile is logged on the ``MLflow`` server as a html file.
            Ignored when ``log_experiment`` is False.


        log_data: bool, default = False
            When set to True, dataset is logged on the ``MLflow`` server as a csv file.
            Ignored when ``log_experiment`` is False.


        verbose: bool, default = True
            When set to False, Information grid is not printed.


        memory: str, bool or Memory, default=True
            Used to cache the fitted transformers of the pipeline.
                If False: No caching is performed.
                If True: A default temp directory is used.
                If str: Path to the caching directory.

        profile: bool, default = False
            When set to True, an interactive EDA report is displayed.


        profile_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the ProfileReport method used
            to create the EDA report. Ignored if ``profile`` is False.


        Returns:
            Global variables that can be changed using the ``set_config`` function.

        """




        self._register_setup_params(dict(locals()))

        if (data is None and data_func is None) or (
                data is not None and data_func is not None
        ):
            raise ValueError("One and only one of data and data_func must be set")

        # No extra code above this line
        # Setup initialization ===================================== >>

        runtime_start = time.time()

        if data_func is not None:
            data = data_func()

        # Define parameter attrs
        self.fold_shuffle_param = fold_shuffle
        self.fold_groups_param = fold_groups

        self._initialize_setup(
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            html=html,
            session_id=session_id,
            system_log=system_log,
            log_experiment=log_experiment,
            experiment_name=experiment_name,
            memory=memory,
            verbose=verbose,
        )

        # Prepare experiment specific params ======================= >>

        self.log_plots_param = log_plots
        if self.log_plots_param is True:
            self.log_plots_param = self._get_default_plots_to_log()
        elif isinstance(self.log_plots_param, list):
            for i in self.log_plots_param:
                if i not in self._available_plots:
                    raise ValueError(
                        f"Invalid value for log_plots '{i}'. Possible values "
                        f"are: {', '.join(self._available_plots.keys())}."
                    )

        # Check transform_target_method
        allowed_transform_target_method = ["quantile", "yeo-johnson"]
        if transform_target_method not in allowed_transform_target_method:
            raise ValueError(
                "Invalid value for the transform_target_method parameter. "
                f"Choose from: {', '.join(allowed_transform_target_method)}."
            )
        self.transform_target_param = transform_target
        self.transform_target_method = transform_target_method

        # Set up data ============================================== >>
        # _, self.data_targets = sa_data.get_x_y(data_frame=data,
        #                                        attr_labels=[target, duration],
        #                                        pos_label=target,
        #                                        survival=True)

        # Patch: set self.X to feature columns (drop event/time) before inferring column types
        
        self.target_param = 'event'
        target_event = target[0]
        target_time = target[1]


        data.rename(columns={target_event: "event", target_time: "time"}, inplace=True)
        print("[DEBUG] Data columns after renaming:", data.columns.tolist())
        self.data = self._prepare_dataset(X=data, y=self.target_param)
        print("[DEBUG] Data columns after preparation:", self.data.columns.tolist())

        self.time_range_percentiles = time_range_percentiles

        lower, upper = np.percentile(data["time"], self.time_range_percentiles)
        
        # print(f"Time range: {lower} - {upper}")
        self.time_range = np.arange(lower, upper)

        
        # For survival analysis, we need to keep the time column in the features
        # Only drop the event column (target)
        self._X = self.data.drop(columns=[self.target_param])



        self._prepare_column_types(
            ordinal_features=ordinal_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            date_features=date_features,
            text_features=text_features,
            ignore_features=ignore_features,
            keep_features=keep_features,
            feature_df=self._X
        )

        self.logger.info("Preparing data for training and testing...")

        self._prepare_train_test(
            train_size=train_size,
            test_data=test_data,
            data_split_stratify=data_split_stratify,
            data_split_shuffle=data_split_shuffle,
        )
        self._prepare_folds(
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
        )

        # Preprocessing ============================================ >>
        pipeline_kwargs = {'survival_train': 'self.survival_train', 'time_range': 'self.time_range'}
        # Initialize empty pipeline
        self.pipeline = InternalPipeline(
            steps=[("placeholder", None)],
            memory=self.memory,
            pipeline_kwargs = pipeline_kwargs,

        )

        if preprocess:
            self.logger.info("Preparing preprocessing pipeline...")

            # Encode the target column
            # if self.y.dtype.kind not in "ifu":
            #     self._encode_target_column()

            # Power transform the target to be more Gaussian-like
            if transform_target:
                self._target_transformation(transform_target_method)

            # Convert date feature to numerical values
            if self._fxs["Date"]:
                self._date_feature_engineering()

            # Impute missing values
            if imputation_type == "simple":
                self._simple_imputation(numeric_imputation, categorical_imputation)
            elif imputation_type == "iterative":
                self._iterative_imputation(
                    iterative_imputation_iters=iterative_imputation_iters,
                    numeric_iterative_imputer=numeric_iterative_imputer,
                    categorical_iterative_imputer=categorical_iterative_imputer,
                )
            elif imputation_type is not None:
                raise ValueError(
                    "Invalid value for the imputation_type parameter, got "
                    f"{imputation_type}. Possible values are: simple, iterative."
                )

            # Convert text features to meaningful vectors
            if self._fxs["Text"]:
                self._text_embedding(text_features_method)

            # Encode non-numerical features
            if self._fxs["Ordinal"] or self._fxs["Categorical"]:
                self._encoding(max_encoding_ohe, encoding_method)

            # Create polynomial features from the existing ones
            if polynomial_features:
                self._polynomial_features(polynomial_degree)

            # Drop features with too low variance
            if low_variance_threshold:
                self._low_variance(low_variance_threshold)

            # Drop features that are collinear with other features
            if remove_multicollinearity:
                self._remove_multicollinearity(multicollinearity_threshold)

            # Bin numerical features to 5 clusters
            if bin_numeric_features:
                self._bin_numerical_features(bin_numeric_features)

            # Remove outliers from the dataset
            if remove_outliers:
                self._remove_outliers(outliers_method, outliers_threshold)

            # Power transform the data to be more Gaussian-like
            if transformation:
                self._transformation(transformation_method)

            # Scale the features
            if normalize:
                self._normalization(normalize_method)

            # Apply Principal Component Analysis
            if pca:
                self._pca(pca_method, pca_components)

            # Select relevant features
            if feature_selection:
                self._feature_selection(
                    feature_selection_method=feature_selection_method,
                    feature_selection_estimator=feature_selection_estimator,
                    n_features_to_select=n_features_to_select,
                )

        # Add custom transformers to the pipeline
        if custom_pipeline:
            self._add_custom_pipeline(custom_pipeline)

        # Remove placeholder step
        if len(self.pipeline) > 1:
            self.pipeline.steps.pop(0)

        print("[DEBUG] Attributes before fitting:")
        for attr_name, attr_value in self.__dict__.items():
            print(f"  {attr_name}: {attr_value}")

        print("[DEBUG] Pipeline steps before fitting:", self.pipeline.steps)
        print("X_train head:", self.X_train)
        print("y_train head:", self.y_train)
        self.pipeline.fit(self.X_train, self.y_train)
        

        self.logger.info(f"Finished creating preprocessing pipeline.")
        self.logger.info(f"Pipeline: {self.pipeline}")
        print("[DEBUG] Pipeline steps:", self.pipeline.steps)

        # Final display ============================================ >>

        self.logger.info("Creating final display dataframe.")
        print("[DEBUG] X_transformed type:", type(self.X_train_transformed))
        print("[DEBUG] X_transformed columns:", None if self.X_train_transformed is None else self.X_train_transformed.columns.tolist())
        print("[DEBUG] X_transformed shape:", None if self.X_train_transformed is None else self.X_train_transformed.shape)
        print("[DEBUG] Columns in X_transformed:", None if self.X_transformed is None else self.X_transformed.columns.tolist())
        print("[DEBUG] y_transformed type:", type(self.y_transformed))
        print("[DEBUG] y_transformed:", self.y_transformed)
        print("[DEBUG] target argument:", target)
        if self.X_transformed is None or self.y_transformed is None:
            raise ValueError("X_transformed or y_transformed is None. Check preprocessing steps.")
        # if 'time' not in self.X_transformed.columns:
        #     raise ValueError("'time' column not found in X_transformed. Check your target and keep_features settings. Columns: {}".format(self.X_transformed.columns.tolist()))
        self.survival_train = Surv.from_arrays(event=self.y_transformed, time=self.X_transformed['time'])

        container = []
        container.append(["Session id", self.seed])
        container.append(["Target", self.target_param])
        container.append(["Target type", "Survival Analysis"])
        container.append(["Data shape", self.dataset_transformed.shape])
        container.append(["Train data shape", self.train_transformed.shape])
        container.append(["Test data shape", self.test_transformed.shape])
        for fx, cols in self._fxs.items():
            if len(cols) > 0:
                container.append([f"{fx} features", len(cols)])
        if self.data.isna().sum().sum():
            n_nans = 100 * self.data.isna().any(axis=1).sum() / len(self.data)
            container.append(["Rows with missing values", f"{round(n_nans, 1)}%"])
        if preprocess:
            container.append(["Preprocess", preprocess])
            container.append(["Imputation type", imputation_type])
            if imputation_type == "simple":
                container.append(["Numeric imputation", numeric_imputation])
                container.append(["Categorical imputation", categorical_imputation])
            else:
                if isinstance(numeric_iterative_imputer, str):
                    num_imputer = numeric_iterative_imputer
                else:
                    num_imputer = numeric_iterative_imputer.__class__.__name__

                if isinstance(categorical_iterative_imputer, str):
                    cat_imputer = categorical_iterative_imputer
                else:
                    cat_imputer = categorical_iterative_imputer.__class__.__name__

                container.append(
                    ["Iterative imputation iterations", iterative_imputation_iters]
                )
                container.append(["Numeric iterative imputer", num_imputer])
                container.append(["Categorical iterative imputer", cat_imputer])
            if self._fxs["Text"]:
                container.append(
                    ["Text features embedding method", text_features_method]
                )
            if self._fxs["Categorical"]:
                container.append(["Maximum one-hot encoding", max_encoding_ohe])
                container.append(["Encoding method", encoding_method])
            if polynomial_features:
                container.append(["Polynomial features", polynomial_features])
                container.append(["Polynomial degree", polynomial_degree])
            if low_variance_threshold:
                container.append(["Low variance threshold", low_variance_threshold])
            if remove_multicollinearity:
                container.append(["Remove multicollinearity", remove_multicollinearity])
                container.append(
                    ["Multicollinearity threshold", multicollinearity_threshold]
                )
            if remove_outliers:
                container.append(["Remove outliers", remove_outliers])
                container.append(["Outliers threshold", outliers_threshold])
            if transformation:
                container.append(["Transformation", transformation])
                container.append(["Transformation method", transformation_method])
            if normalize:
                container.append(["Normalize", normalize])
                container.append(["Normalize method", normalize_method])
            if pca:
                container.append(["PCA", pca])
                container.append(["PCA method", pca_method])
                container.append(["PCA components", pca_components])
            if feature_selection:
                container.append(["Feature selection", feature_selection])
                container.append(["Feature selection method", feature_selection_method])
                container.append(
                    ["Feature selection estimator", feature_selection_estimator]
                )
                container.append(["Number of features selected", n_features_to_select])
            if transform_target:
                container.append(["Transform target", transform_target])
                container.append(["Transform target method", transform_target_method])
            if custom_pipeline:
                container.append(["Custom pipeline", "Yes"])
            container.append(["Fold Generator", self.fold_generator.__class__.__name__])
            container.append(["Fold Number", fold])
            container.append(["CPU Jobs", self.n_jobs_param])
            container.append(["Use GPU", self.gpu_param])
            container.append(["Log Experiment", self.logging_param])
            container.append(["Experiment Name", self.exp_name_log])
            container.append(["USI", self.USI])

        self.display_container = [
            pd.DataFrame(container, columns=["Description", "Value"])
        ]
        self.logger.info(f"Setup display_container: {self.display_container[0]}")
        display = CommonDisplay(
            verbose=self.verbose,
            html_param=self.html_param,
        )
        if self.verbose:
            pd.set_option("display.max_rows", 100)
            display.display(self.display_container[0].style.apply(highlight_setup))
            pd.reset_option("display.max_rows")  # Reset option

        # Wrap-up ================================================== >>

        # Create a profile report
        self._profile(profile, profile_kwargs)

        # Define models and metrics
        self._all_models, self._all_models_internal = self._get_models()
        self._all_metrics = self._get_metrics()

        runtime = np.array(time.time() - runtime_start).round(2)
        self._set_up_logging(
            runtime,
            log_data,
            log_profile,
            experiment_custom_tags=experiment_custom_tags,
        )

        self._setup_ran = True
        self.logger.info(f"setup() successfully completed in {runtime}s...............")
        # self._ml_usecase = MLUsecase.TIME_SERIES
        return self

    def compare_models(
            self,
            include: Optional[List[Union[str, Any]]] = None,
            exclude: Optional[List[str]] = None,
            fold: Optional[Union[int, Any]] = None,
            round: int = 4,
            cross_validation: bool = True,
            sort: str = "cindex",
            n_select: int = 1,
            budget_time: Optional[float] = None,
            turbo: bool = True,
            errors: str = "raise",
            fit_kwargs: Optional[dict] = None,
            groups: Optional[Union[str, Any]] = None,
            experiment_custom_tags: Optional[Dict[str, Any]] = None,
            verbose: bool = True,
            parallel: Optional[ParallelBackend] = None,
    ):

        """
        This function trains and evaluates performance of all estimators available in the
        model library using cross validation. The output of this function is a score grid
        with average cross validated scores. Metrics evaluated during CV can be accessed
        using the ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.


        Example
        --------
        >>> from pycaret_local.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret_local.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> best_model = compare_models()


        include: list of str or scikit-learn compatible object, default = None
            To train and evaluate select models, list containing model ID or scikit-learn
            compatible object can be passed in include param. To see a list of all models
            available in the model library use the ``models`` function.


        exclude: list of str, default = None
            To omit certain models from training and evaluation, pass a list containing
            model id in the exclude parameter. To see a list of all models available
            in the model library use the ``models`` function.


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        cross_validation: bool, default = True
            When set to False, metrics are evaluated on holdout set. ``fold`` param
            is ignored when cross_validation is set to False.


        sort: str, default = 'R2'
            The sort order of the score grid. It also accepts custom metrics that are
            added through the ``add_metric`` function.


        n_select: int, default = 1
            Number of top_n models to return. For example, to select top 3 models use
            n_select = 3.


        budget_time: int or float, default = None
            If not None, will terminate execution of the function after budget_time
            minutes have passed and return results up to that point.


        turbo: bool, default = True
            When set to True, it excludes estimators with longer training times. To
            see which algorithms are excluded use the ``models`` function.


        errors: str, default = 'ignore'
            When set to 'ignore', will skip the model with exceptions and continue.
            If 'raise', will break the function when exceptions are raised.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when 'GroupKFold' is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in the training dataset. When string is passed, it is interpreted
            as the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        parallel: pycaret_local.internal.parallel.parallel_backend.ParallelBackend, default = None
            A ParallelBackend instance. For example if you have a SparkSession ``session``,
            you can use ``FugueBackend(session)`` to make this function running using
            Spark. For more details, see
            :class:`~pycaret_local.parallel.fugue_backend.FugueBackend`


        Returns:
            Trained model or list of trained models, depending on the ``n_select`` param.


        Warnings
        --------
        - Changing turbo parameter to False may result in very high training times with
        datasets exceeding 10,000 rows.

        - No models are logged in ``MLFlow`` when ``cross_validation`` parameter is False.

        """

        caller_params = dict(locals())

        # No extra code above this line

        return super().compare_models(
            include=include,
            exclude=exclude,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            sort=sort,
            n_select=n_select,
            budget_time=budget_time,
            turbo=turbo,
            errors=errors,
            fit_kwargs=fit_kwargs,
            groups=groups,
            experiment_custom_tags=experiment_custom_tags,
            verbose=verbose,
            parallel=parallel,
            caller_params=caller_params,
        )

    def _create_model_without_cv(
        self,
        model,
        data_X,
        data_y,
        fit_kwargs,
        round,
        predict,
        system,
        display: CommonDisplay,
        return_train_score: bool = False,
    ):
        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info("Cross validation set to False")

            self.logger.info("Fitting Model")
            model_fit_start = time.time()
            with redirect_output(self.logger):
                pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            display.move_progress()
            new_data = pd.concat([data_X, data_y], axis=1)
            new_data2 = data_X.assign(event=data_y)
            if predict:
                if return_train_score:
                    # call class explicitly to get access to preprocess arg
                    # in subclasses
                    self.predict_model(
                        estimator=pipeline_with_model,
                        data=pd.concat([data_X, data_y], axis=1),
                        verbose=False,
                    )
                    train_results = self.pull(pop=True).drop("Model", axis=1)
                    train_results.index = ["Train"]
                else:
                    train_results = None

                self.predict_model(pipeline_with_model, verbose=False)
                model_results = self.pull(pop=True).drop("Model", axis=1)
                model_results.index = ["Test"]
                if train_results is not None:
                    model_results = pd.concat([model_results, train_results])

                self.display_container.append(model_results)

                model_results = model_results.style.format(precision=round)

                if system:
                    display.display(model_results)

                self.logger.info(f"display_container: {len(self.display_container)}")

        return model, model_fit_time

    def _create_model_with_cv(
        self,
        model,
        data_X,
        data_y,
        fit_kwargs,
        round,
        cv,
        groups,
        metrics,
        refit,
        system,
        display,
        return_train_score: bool = False,
    ):
        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(
            1,
            f"Fitting {self._get_cv_n_folds(cv, data_X, y=data_y, groups=groups)} Folds",
        )
        """
        MONITOR UPDATE ENDS
        """

        from sklearn.model_selection import cross_validate

        metrics_dict = dict([(k, v.scorer) for k, v in metrics.items()])

        self.logger.info("Starting cross validation")

        n_jobs = self._gpu_n_jobs_param
        from sklearn.gaussian_process import (
            GaussianProcessClassifier,
            GaussianProcessRegressor,
        )
        from sklearn.model_selection import GridSearchCV
        # special case to prevent running out of memory
        if isinstance(model, (GaussianProcessClassifier, GaussianProcessRegressor)):
            n_jobs = 1

        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info(f"Cross validating with {cv}, n_jobs={n_jobs}")

            model_fit_start = time.time()
            refit = True
            # with redirect_output(self.logger):
            scores = cross_validate(
                pipeline_with_model,
                data_X,
                data_y,
                cv=cv,
                groups=groups,
                scoring=metrics_dict,
                fit_params=fit_kwargs,
                n_jobs=n_jobs,
                return_train_score=return_train_score,
                error_score='raise',
                verbose=False,
            )
            model_fit_end = time.time()
            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
            # scores = gcv.cv_results_
            score_dict = {}
            for k, v in metrics.items():
                score_dict[v.display_name] = []
                if return_train_score:
                    train_score = scores[f"train_{k}"] * (
                        1 if v.greater_is_better else -1
                    )
                    train_score = train_score.tolist()
                    score_dict[v.display_name] = train_score
                test_score = scores[f"test_{k}"] * (1 if v.greater_is_better else -1)
                test_score = test_score.tolist()
                score_dict[v.display_name] += test_score

            self.logger.info("Calculating mean and std")

            avgs_dict = {}
            for k, v in metrics.items():
                avgs_dict[v.display_name] = []
                if return_train_score:
                    train_score = scores[f"train_{k}"] * (
                        1 if v.greater_is_better else -1
                    )
                    train_score = train_score.tolist()
                    avgs_dict[v.display_name] = [
                        np.mean(train_score),
                        np.std(train_score),
                    ]
                test_score = scores[f"test_{k}"] * (1 if v.greater_is_better else -1)
                test_score = test_score.tolist()
                avgs_dict[v.display_name] += [np.mean(test_score), np.std(test_score)]

            display.move_progress()

            self.logger.info("Creating metrics dataframe")

            fold = cv.n_splits

            if return_train_score:
                model_results = pd.DataFrame(
                    {
                        "Split": ["CV-Train"] * fold
                        + ["CV-Val"] * fold
                        + ["CV-Train"] * 2
                        + ["CV-Val"] * 2,
                        "Fold": np.arange(fold).tolist()
                        + np.arange(fold).tolist()
                        + ["Mean", "Std"] * 2,
                    }
                )
            else:
                model_results = pd.DataFrame(
                    {
                        "Fold": np.arange(fold).tolist() + ["Mean", "Std"],
                    }
                )

            model_scores = pd.concat(
                [pd.DataFrame(score_dict), pd.DataFrame(avgs_dict)]
            ).reset_index(drop=True)

            model_results = pd.concat([model_results, model_scores], axis=1)
            model_results.set_index(
                self._get_return_train_score_columns_for_display(return_train_score),
                inplace=True,
            )

            if refit:
                # refitting the model on complete X_train, y_train
                display.update_monitor(1, "Finalizing Model")
                model_fit_start = time.time()
                self.logger.info("Finalizing model")
                with redirect_output(self.logger):
                    pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
                    model_fit_end = time.time()

                # calculating metrics on predictions of complete train dataset
                if return_train_score:
                    # call class explicitly to get access to preprocess arg
                    # in subclasses
                    self.predict_model(
                        pipeline_with_model,
                        data=pd.concat([data_X, data_y], axis=1),
                        verbose=False,
                    )
                    metrics = self.pull(pop=True).drop("Model", axis=1)
                    df_score = pd.DataFrame({"Split": ["Train"], "Fold": [None]})
                    df_score = pd.concat([df_score, metrics], axis=1)
                    df_score.set_index(["Split", "Fold"], inplace=True)

                    # concatenating train results to cross-validation socre dataframe
                    model_results = pd.concat([model_results, df_score])

                model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
            else:
                model_fit_time /= self._get_cv_n_folds(
                    cv, data_X, y=data_y, groups=groups
                )

        model_results = model_results.round(round)

        return model, model_fit_time, model_results, avgs_dict

    def _create_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        predict: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        refit: bool = True,
        probability_threshold: Optional[float] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        system: bool = True,
        add_to_model_list: bool = True,
        X_train_data: Optional[pd.DataFrame] = None,  # added in pycaret_local==2.2.0
        y_train_data: Optional[pd.DataFrame] = None,  # added in pycaret_local==2.2.0
        metrics=None,
        display: Optional[CommonDisplay] = None,  # added in pycaret_local==2.2.0
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:

        """
        Internal version of ``create_model`` with private arguments.
        """
        self._check_setup_ran()

        function_params_str = ", ".join(
            [
                f"{k}={v}"
                for k, v in locals().items()
                if k not in ("X_train_data", "y_train_data")
            ]
        )

        self.logger.info("Initializing create_model()")
        self.logger.info(f"create_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        available_estimators = set(self._all_models_internal.keys())

        if not fit_kwargs:
            fit_kwargs = {}

        # only raise exception of estimator is of type string.
        if isinstance(estimator, str):
            if estimator not in available_estimators:
                raise ValueError(
                    f"Estimator {estimator} not available. Please see docstring for list of available estimators."
                )
        elif not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        # checking round parameter
        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        # checking verbose parameter
        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )

        # checking system parameter
        if type(system) is not bool:
            raise TypeError("System parameter can only take argument as True or False.")

        # checking cross_validation parameter
        if type(cross_validation) is not bool:
            raise TypeError(
                "cross_validation parameter can only take argument as True or False."
            )

        # checking return_train_score parameter
        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )

        """

        ERROR HANDLING ENDS HERE

        """

        groups = self._get_groups(groups, data=X_train_data)

        if not display:
            progress_args = {"max": 4}
            timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
            monitor_rows = [
                ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
                [
                    "Status",
                    ". . . . . . . . . . . . . . . . . .",
                    "Loading Dependencies",
                ],
                [
                    "Estimator",
                    ". . . . . . . . . . . . . . . . . .",
                    "Compiling Library",
                ],
            ]
            display = CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                monitor_rows=monitor_rows,
            )

        self.logger.info("Importing libraries")

        # general dependencies

        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")

        # Storing X_train and y_train in data_X and data_y parameter
        if self._ml_usecase != MLUsecase.TIME_SERIES:
            data_X = self.X_train if X_train_data is None else X_train_data.copy()
            data_y = self.y_train if y_train_data is None else y_train_data.copy()

            data_X.reset_index(drop=True, inplace=True)
            data_y.reset_index(drop=True, inplace=True)
        else:
            if X_train_data is not None:
                data_X = X_train_data.copy()
            else:
                if self.X_train is None:
                    data_X = None
                else:
                    data_X = self.X_train
            data_y = self.y_train if y_train_data is None else y_train_data.copy()

        if metrics is None:
            metrics = self._all_metrics

        display.move_progress()

        self.logger.info("Defining folds")

        # cross validation setup starts here
        if self._ml_usecase == MLUsecase.TIME_SERIES:
            cv = self.get_fold_generator(fold=fold)
        else:
            cv = self._get_cv_splitter(fold)

        if self._ml_usecase == MLUsecase.TIME_SERIES:
            # Add forecast horizon if use case is Time Series
            fit_kwargs = self.update_fit_kwargs_with_fh_from_cv(
                fit_kwargs=fit_kwargs, cv=cv
            )

        self.logger.info("Declaring metric variables")

        """
        MONITOR UPDATE STARTS
        """
        display.update_monitor(1, "Selecting Estimator")
        """
        MONITOR UPDATE ENDS
        """

        self.logger.info("Importing untrained model")

        if isinstance(estimator, str) and estimator in available_estimators:
            model_definition = self._all_models_internal[estimator]
            model_args = model_definition.args
            model_args = {**model_args, **kwargs}
            model = model_definition.class_def(**model_args)
            full_name = model_definition.name
        else:
            self.logger.info("Declaring custom model")

            model = clone(estimator)
            model.set_params(**kwargs)

            full_name = self._get_model_name(model)

        # workaround for an issue with set_params in cuML
        model = clone(model)

        display.update_monitor(2, full_name)

        if (
            probability_threshold
            and self._ml_usecase == MLUsecase.CLASSIFICATION
            and not self._is_multiclass
        ):
            if not isinstance(model, CustomProbabilityThresholdClassifier):
                model = CustomProbabilityThresholdClassifier(
                    classifier=model,
                    probability_threshold=probability_threshold,
                )
            elif probability_threshold is not None:
                model.set_params(probability_threshold=probability_threshold)
        self.logger.info(f"{full_name} Imported successfully")

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """
        if not cross_validation:
            display.update_monitor(1, f"Fitting {str(full_name)}")
        else:
            display.update_monitor(1, "Initializing CV")

        """
        MONITOR UPDATE ENDS
        """

        if not cross_validation:
            model, model_fit_time = self._create_model_without_cv(
                model,
                data_X,
                data_y,
                fit_kwargs,
                round,
                predict,
                system,
                display,
                return_train_score=return_train_score,
            )

            display.move_progress()

            self.logger.info(str(model))
            self.logger.info(
                "create_model() successfully completed......................................"
            )

            gc.collect()

            if not system:
                return model, model_fit_time
            return model

        model, model_fit_time, model_results, _ = self._create_model_with_cv(
            model,
            data_X,
            data_y,
            fit_kwargs,
            round,
            cv,
            groups,
            metrics,
            refit,
            system,
            display,
            return_train_score=return_train_score,
        )

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)

        # dashboard logging
        if self.logging_param and system and refit:
            indices = self._get_return_train_score_indices_for_logging(
                return_train_score
            )
            avgs_dict_log = {k: v for k, v in model_results.loc[indices].items()}

            self._log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="create_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                experiment_custom_tags=experiment_custom_tags,
                display=display,
            )

        display.move_progress()

        self.logger.info("Uploading results into container")

        if not self._ml_usecase == MLUsecase.TIME_SERIES:
            model_results.drop("cutoff", axis=1, inplace=True, errors="ignore")

        self.display_container.append(model_results)

        # storing results in master_model_container
        if add_to_model_list:
            self.logger.info("Uploading model into container now")
            self.master_model_container.append(
                {"model": model, "scores": model_results, "cv": cv}
            )

        # yellow the mean
        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        if system:
            display.display(model_results)

        self.logger.info(f"master_model_container: {len(self.master_model_container)}")
        self.logger.info(f"display_container: {len(self.display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "create_model() successfully completed......................................"
        )
        gc.collect()

        if not system:
            return model, model_fit_time

        return model

    def create_model(
            self,
            estimator: Union[str, Any],
            fold: Optional[Union[int, Any]] = None,
            round: int = 4,
            cross_validation: bool = True,
            predict: bool = True,
            fit_kwargs: Optional[dict] = None,
            groups: Optional[Union[str, Any]] = None,
            refit: bool = True,
            probability_threshold: Optional[float] = None,
            experiment_custom_tags: Optional[Dict[str, Any]] = None,
            verbose: bool = True,
            return_train_score: bool = False,
            **kwargs,
    ):

        """
        This function trains and evaluates the performance of a given estimator
        using cross validation. The output of this function is a score grid with
        CV scores by fold. Metrics evaluated during CV can be accessed using the
        ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function. All the available models
        can be accessed using the ``models`` function.


        Example
        -------
        >>> from pycaret_local.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret_local.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')


        estimator: str or scikit-learn compatible object
            ID of an estimator available in model library or pass an untrained
            model object consistent with scikit-learn API. Estimators available
            in the model library (ID - Name):

            * 'lr' - Linear Regression
            * 'lasso' - Lasso Regression
            * 'ridge' - Ridge Regression
            * 'en' - Elastic Net
            * 'lar' - Least Angle Regression
            * 'llar' - Lasso Least Angle Regression
            * 'omp' - Orthogonal Matching Pursuit
            * 'br' - Bayesian Ridge
            * 'ard' - Automatic Relevance Determination
            * 'par' - Passive Aggressive Regressor
            * 'ransac' - Random Sample Consensus
            * 'tr' - TheilSen Regressor
            * 'huber' - Huber Regressor
            * 'kr' - Kernel Ridge
            * 'svm' - Support Vector Regression
            * 'knn' - K Neighbors Regressor
            * 'dt' - Decision Tree Regressor
            * 'rf' - Random Forest Regressor
            * 'et' - Extra Trees Regressor
            * 'ada' - AdaBoost Regressor
            * 'gbr' - Gradient Boosting Regressor
            * 'mlp' - MLP Regressor
            * 'xgboost' - Extreme Gradient Boosting
            * 'lightgbm' - Light Gradient Boosting Machine
            * 'catboost' - CatBoost Regressor


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        cross_validation: bool, default = True
            When set to False, metrics are evaluated on holdout set. ``fold`` param
            is ignored when cross_validation is set to False.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        experiment_custom_tags: dict, default = None
            Dictionary of tag_name: String -> value: (String, but will be string-ified
            if not) passed to the mlflow.set_tags to add new custom tags for the experiment.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        **kwargs:
            Additional keyword arguments to pass to the estimator.


        Returns:
            Trained Model


        Warnings
        --------
        - Models are not logged on the ``MLFlow`` server when ``cross_validation`` param
        is set to False.

        """

        assert not any(
            x
            in (
                "system",
                "add_to_model_list",
                "X_train_data",
                "y_train_data",
                "metrics",
            )
            for x in kwargs
        )
        return self._create_model(
            estimator=estimator,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            predict=predict,
            fit_kwargs=fit_kwargs,
            groups=groups,
            refit=refit,
            probability_threshold=probability_threshold,
            experiment_custom_tags=experiment_custom_tags,
            verbose=verbose,
            return_train_score=return_train_score,
            **kwargs,
        )

    def predict_model(
            self,
            estimator,
            data: Optional[pd.DataFrame] = None,
            drift_report: bool = False,
            round: int = 4,
            verbose: bool = True,
    ) -> pd.DataFrame:

        """
        This function predicts ``Label`` using a trained model. When ``data`` is
        None, it predicts label on the holdout set.


        Example
        -------
        >>> from pycaret_local.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret_local.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> pred_holdout = predict_model(lr)
        >>> pred_unseen = predict_model(lr, data = unseen_dataframe)


        estimator: scikit-learn compatible object
            Trained model object


        data : pandas.DataFrame
            Shape (n_samples, n_features). All features used during training
            must be available in the unseen dataset.


        drift_report: bool, default = False
            When set to True, interactive drift report is generated on test set
            with the evidently library.


        round: int, default = 4
            Number of decimal places to round predictions to.


        verbose: bool, default = True
            When set to False, holdout score grid is not printed.


        Returns:
            pandas.DataFrame


        Warnings
        --------
        - The behavior of the ``predict_model`` is changed in version 2.1 without backward
        compatibility. As such, the pipelines trained using the version (<= 2.0), may not
        work for inference with version >= 2.1. You can either retrain your models with a
        newer version or downgrade the version for inference.


        """
        if data is not None:
            for col in self.target_param:
                if col in data.columns:
                    data.drop(columns=col)

        return self._predict_model(estimator=estimator,
                                     data=data,
                                     probability_threshold=None,
                                     encoded_labels=True,
                                     drift_report=drift_report,
                                     round=round,
                                     verbose=verbose)

    def _predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,  # added in pycaret_local==2.1.0
        raw_score: bool = False,
        drift_report: bool = False,
        round: int = 4,  # added in pycaret_local==2.2.0
        verbose: bool = True,
        ml_usecase: Optional[MLUsecase] = None,
        preprocess: Union[bool, str] = True,
    ) -> pd.DataFrame:

        """
        This function is used to predict label and probability score on the new dataset
        using a trained estimator. New unseen data can be passed to data parameter as pandas
        Dataframe. If data is not passed, the test / hold-out set separated at the time of
        setup() is used to generate predictions.

        Example
        -------
        >>> from pycaret_local.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> lr_predictions_holdout = predict_model(lr)

        Parameters
        ----------
        estimator : object, default = none
            A trained model object / pipeline should be passed as an estimator.

        data : pandas.DataFrame
            Shape (n_samples, n_features) where n_samples is the number of samples
            and n_features is the number of features. All features used during training
            must be present in the new dataset.

        probability_threshold : float, default = None
            Threshold used to convert probability values into binary outcome. By default
            the probability threshold for all binary classifiers is 0.5 (50%). This can be
            changed using probability_threshold param.

        encoded_labels: Boolean, default = False
            If True, will return labels encoded as an integer.

        raw_score: bool, default = False
            When set to True, scores for all labels will be returned.

        round: integer, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.

        verbose: bool, default = True
            Holdout score grid is not printed when verbose is set to False.

        preprocess: bool or 'features', default = True
            Whether to preprocess unseen data. If 'features', will not
            preprocess labels.

        Returns
        -------
        Predictions
            Predictions (Label and Score) column attached to the original dataset
            and returned as pandas dataframe.

        score_grid
            A table containing the scoring metrics on hold-out / test set.

        Warnings
        --------
        - The behavior of the predict_model is changed in version 2.1 without backward compatibility.
        As such, the pipelines trained using the version (<= 2.0), may not work for inference
        with version >= 2.1. You can either retrain your models with a newer version or downgrade
        the version for inference.

        """

        def replace_labels_in_column(pipeline, labels: pd.Series) -> pd.Series:
            # Check if there is a LabelEncoder in the pipeline
            name = labels.name
            index = labels.index
            le = get_label_encoder(pipeline)
            if le:
                return pd.Series(le.inverse_transform(labels), name=name, index=index)
            else:
                return labels

        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "data"]
        )

        self.logger.info("Initializing predict_model()")
        self.logger.info(f"predict_model({function_params_str})")

        self.logger.info("Checking exceptions")

        """
        exception checking starts here
        """

        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        if data is None and not self._setup_ran:
            raise ValueError(
                "data parameter may not be None without running setup() first."
            )

        if probability_threshold is not None:
            # probability_threshold allowed types
            allowed_types = [int, float]
            if (
                type(probability_threshold) not in allowed_types
                or probability_threshold > 1
                or probability_threshold < 0
            ):
                raise TypeError(
                    "probability_threshold parameter only accepts value between 0 to 1."
                )

        """
        exception checking ends here
        """

        self.logger.info("Preloading libraries")

        try:
            np.random.seed(self.seed)
            display = CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
            )
        except:
            display = CommonDisplay(
                verbose=False,
                html_param=False,
            )

        if isinstance(estimator, InternalPipeline):
            if not hasattr(estimator, "feature_names_in_"):
                raise ValueError(
                    "If estimator is a Pipeline, it must implement `feature_names_in_`."
                )
            pipeline = estimator
            # Temporarily remove final estimator so it's not used for transform
            final_step = pipeline.steps[-1]
            estimator = final_step[-1]
            pipeline.steps = pipeline.steps[:-1]
        elif not self._setup_ran:
            raise ValueError(
                "If estimator is not a Pipeline, you must run setup() first."
            )
        else:
            pipeline = self.pipeline
            final_step = None

        X_columns = pipeline.feature_names_in_[:-1]
        y_name = pipeline.feature_names_in_[-1]
        # y_test_ = None
        if data is None:
            X_test_, y_test_ = self.X_test_transformed, self.y_test_transformed
        else:
            if y_name in data.columns:
                data = self._prepare_dataset(data, y_name)
                target = data[y_name]
                data = data.drop(y_name, axis=1)
            else:
                data = self._prepare_dataset(data)
                target = None
            data = data[X_columns]  # Ignore all columns but the originals
            if preprocess:
                X_test_ = pipeline.transform(
                    X=data,
                    y=(target if preprocess != "features" else None),
                )
                if final_step:
                    pipeline.steps.append(final_step)

                if isinstance(X_test_, tuple):
                    X_test_, y_test_ = X_test_
                elif target is not None:
                    y_test_ = target
            else:
                X_test_ = data
                y_test_ = target

        # generate drift report
        if drift_report:
            _check_soft_dependencies("evidently", extra="mlops", severity="error")
            from evidently.dashboard import Dashboard
            from evidently.pipeline.column_mapping import ColumnMapping
            from evidently.tabs import CatTargetDriftTab, DataDriftTab

            column_mapping = ColumnMapping()
            column_mapping.target = self.target_param
            column_mapping.prediction = None
            column_mapping.datetime = None
            column_mapping.numerical_features = self._fxs["Numeric"]
            column_mapping.categorical_features = self._fxs["Categorical"]
            column_mapping.datetime_features = self._fxs["Date"]

            drift_data = data if data is not None else self.test

            if not y_name in drift_data.columns:
                raise ValueError(
                    f"The dataset must contain a label column {y_name} "
                    "in order to create a drift report."
                )

            dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])
            dashboard.calculate(self.train, drift_data, column_mapping=column_mapping)
            report_name = f"{self._get_model_name(estimator)}_Drift_Report.html"
            dashboard.save(report_name)
            print(f"{report_name} saved successfully.")

        # prediction starts here
        if isinstance(estimator, CustomProbabilityThresholdClassifier):
            if probability_threshold is None:
                probability_threshold = estimator.probability_threshold
            estimator = get_estimator_from_meta_estimator(estimator)

        pred = np.nan_to_num(estimator.predict(X_test_))

        if hasattr(estimator, "predict_survival_function"):
            pred_surv = estimator.predict_survival_function(X_test_)
            if hasattr(estimator, "_predict_cumulative_hazard_function"):
                pred_hazard = estimator._predict_cumulative_hazard_function(X_test_)
            else:
                pred_hazard = estimator.predict_cumulative_hazard_function(X_test_)
            score_func_kwargs = {"pred_surv": pred_surv, "pred_hazard": pred_hazard}

        else:
            score_func_kwargs = {"pred_surv": None, "pred_hazard": None}

        try:
            score = estimator.predict_proba(X_test_)

            if len(np.unique(pred)) <= 2:
                pred_prob = score[:, 1]
            else:
                pred_prob = score

        except:
            score = None
            pred_prob = None

        if probability_threshold is not None and pred_prob is not None:
            try:
                pred = (pred_prob >= probability_threshold).astype(int)
            except:
                pass

        if pred_prob is None:
            pred_prob = pred

        df_score = None
        if y_test_ is not None and self._setup_ran:
            # model name
            full_name = self._get_model_name(estimator)
            metrics_surv = self._calculate_metrics
            metrics = self._calculate_metrics(y_test_,
                                              self.y_train_transformed,
                                              X_test_,
                                              self.X_train_transformed,
                                              pred,
                                              pred_prob,
                                              weights=None,
                                              **score_func_kwargs)  # type: ignore
            df_score = pd.DataFrame(metrics, index=[0])
            df_score.insert(0, "Model", full_name)
            df_score = df_score.round(round)
            display.display(df_score.style.format(precision=round))

        label = pd.DataFrame(pred, columns=["Label"], index=X_test_.index)
        if ml_usecase == MLUsecase.CLASSIFICATION:
            try:
                label["Label"] = label["Label"].astype(int)
            except:
                pass

        if not encoded_labels:
            label["Label"] = replace_labels_in_column(pipeline, label["Label"])
            if y_test_ is not None:
                y_test_ = replace_labels_in_column(pipeline, y_test_)
        old_index = X_test_.index
        X_test_ = pd.concat([X_test_, y_test_, label], axis=1)
        X_test_.index = old_index

        if score is not None:
            pred = pred.astype(int)
            if not raw_score:
                score = [s[pred[i]] for i, s in enumerate(score)]
            try:
                score = pd.DataFrame(score, index=X_test_.index)
                if raw_score:
                    score_columns = pd.Series(
                        range(score.shape[1]), index=X_test_.index
                    )
                    if not encoded_labels:
                        score_columns = replace_labels_in_column(
                            pipeline, score_columns
                        )
                    score.columns = [f"Score_{label}" for label in score_columns]
                else:
                    score.columns = ["Score"]
                score = score.round(round)
                old_index = X_test_.index
                X_test_ = pd.concat((X_test_, score), axis=1)
                X_test_.index = old_index
            except:
                pass

        # store predictions on hold-out in display_container
        if df_score is not None:
            self.display_container.append(df_score)

        gc.collect()
        return X_test_

    def blend_models(
        self,
        estimator_list: list,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        choose_better: bool = False,
        optimize: str = "R2",
        weights: Optional[List[float]] = None,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        return_train_score: bool = False,
    ):

        """
        This function trains a Voting Regressor for select models passed in the
        ``estimator_list`` param. The output of this function is a score grid with
        CV scores by fold. Metrics evaluated during CV can be accessed using the
        ``get_metrics`` function. Custom metrics can be added or removed using
        ``add_metric`` and ``remove_metric`` function.


        Example
        --------
        >>> from pycaret_local.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret_local.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> top3 = compare_models(n_select = 3)
        >>> blender = blend_models(top3)


        estimator_list: list of scikit-learn compatible objects
            List of trained model objects


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        choose_better: bool, default = False
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        optimize: str, default = 'R2'
            Metric to compare for model selection when ``choose_better`` is True.


        weights: list, default = None
            Sequence of weights (float or int) to weight the occurrences of predicted class
            labels (hard voting) or class probabilities before averaging (soft voting). Uses
            uniform weights when None.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        Returns:
            Trained Model


        """

        return super().blend_models(
            estimator_list=estimator_list,
            fold=fold,
            round=round,
            choose_better=choose_better,
            optimize=optimize,
            method="auto",
            weights=weights,
            fit_kwargs=fit_kwargs,
            groups=groups,
            verbose=verbose,
            return_train_score=return_train_score,
        )

    def tune_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        n_iter: int = 10,
        custom_grid: Optional[Union[Dict[str, list], Any]] = None,
        optimize: str = "C-II",
        custom_scorer=None,
        search_library: str = "scikit-learn",
        search_algorithm: Optional[str] = None,
        early_stopping: Any = False,
        early_stopping_max_iters: int = 10,
        choose_better: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        return_tuner: bool = False,
        verbose: bool = True,
        tuner_verbose: Union[int, bool] = True,
        return_train_score: bool = False,
        **kwargs,
    ):

        """
        This function tunes the hyperparameters of a given estimator. The output of
        this function is a score grid with CV scores by fold of the best selected
        model based on ``optimize`` parameter. Metrics evaluated during CV can be
        accessed using the ``get_metrics`` function. Custom metrics can be added
        or removed using ``add_metric`` and ``remove_metric`` function.


        Example
        -------
        >>> from pycaret_local.datasets import get_data
        >>> boston = get_data('boston')
        >>> from pycaret_local.regression import *
        >>> exp_name = setup(data = boston,  target = 'medv')
        >>> lr = create_model('lr')
        >>> tuned_lr = tune_model(lr)


        estimator: scikit-learn compatible object
            Trained model object


        fold: int or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, the CV generator in the ``fold_strategy``
            parameter of the ``setup`` function is used. When an integer is passed,
            it is interpreted as the 'n_splits' parameter of the CV generator in the
            ``setup`` function.


        round: int, default = 4
            Number of decimal places the metrics in the score grid will be rounded to.


        n_iter: int, default = 10
            Number of iterations in the grid search. Increasing 'n_iter' may improve
            model performance but also increases the training time.


        custom_grid: dictionary, default = None
            To define custom search space for hyperparameters, pass a dictionary with
            parameter name and values to be iterated. Custom grids must be in a format
            supported by the defined ``search_library``.


        optimize: str, default = 'R2'
            Metric name to be evaluated for hyperparameter tuning. It also accepts custom
            metrics that are added through the ``add_metric`` function.


        custom_scorer: object, default = None
            custom scoring strategy can be passed to tune hyperparameters of the model.
            It must be created using ``sklearn.make_scorer``. It is equivalent of adding
            custom metric using the ``add_metric`` function and passing the name of the
            custom metric in the ``optimize`` parameter.
            Will be deprecated in future.


        search_library: str, default = 'scikit-learn'
            The search library used for tuning hyperparameters. Possible values:

            - 'scikit-learn' - default, requires no further installation
                https://github.com/scikit-learn/scikit-learn

            - 'scikit-optimize' - ``pip install scikit-optimize``
                https://scikit-optimize.github.io/stable/

            - 'tune-sklearn' - ``pip install tune-sklearn ray[tune]``
                https://github.com/ray-project/tune-sklearn

            - 'optuna' - ``pip install optuna``
                https://optuna.org/


        search_algorithm: str, default = None
            The search algorithm depends on the ``search_library`` parameter.
            Some search algorithms require additional libraries to be installed.
            If None, will use search library-specific default algorithm.

            - 'scikit-learn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search

            - 'scikit-optimize' possible values:
                - 'bayesian' : Bayesian search (default)

            - 'tune-sklearn' possible values:
                - 'random' : random grid search (default)
                - 'grid' : grid search
                - 'bayesian' : ``pip install scikit-optimize``
                - 'hyperopt' : ``pip install hyperopt``
                - 'optuna' : ``pip install optuna``
                - 'bohb' : ``pip install hpbandster ConfigSpace``

            - 'optuna' possible values:
                - 'random' : randomized search
                - 'tpe' : Tree-structured Parzen Estimator search (default)


        early_stopping: bool or str or object, default = False
            Use early stopping to stop fitting to a hyperparameter configuration
            if it performs poorly. Ignored when ``search_library`` is scikit-learn,
            or if the estimator does not have 'partial_fit' attribute. If False or
            None, early stopping will not be used. Can be either an object accepted
            by the search library or one of the following:

            - 'asha' for Asynchronous Successive Halving Algorithm
            - 'hyperband' for Hyperband
            - 'median' for Median Stopping Rule
            - If False or None, early stopping will not be used.


        early_stopping_max_iters: int, default = 10
            Maximum number of epochs to run for each sampled configuration.
            Ignored if ``early_stopping`` is False or None.


        choose_better: bool, default = True
            When set to True, the returned object is always better performing. The
            metric used for comparison is defined by the ``optimize`` parameter.


        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the tuner.


        groups: str or array-like, with shape (n_samples,), default = None
            Optional group labels when GroupKFold is used for the cross validation.
            It takes an array with shape (n_samples, ) where n_samples is the number
            of rows in training dataset. When string is passed, it is interpreted as
            the column name in the dataset containing group labels.


        return_tuner: bool, default = False
            When set to True, will return a tuple of (model, tuner_object).


        verbose: bool, default = True
            Score grid is not printed when verbose is set to False.


        tuner_verbose: bool or in, default = True
            If True or above 0, will print messages from the tuner. Higher values
            print more messages. Ignored when ``verbose`` param is False.


        return_train_score: bool, default = False
            If False, returns the CV Validation scores only.
            If True, returns the CV training scores along with the CV validation scores.
            This is useful when the user wants to do bias-variance tradeoff. A high CV
            training score with a low corresponding CV validation score indicates overfitting.


        **kwargs:
            Additional keyword arguments to pass to the optimizer.


        Returns:
            Trained Model and Optional Tuner Object when ``return_tuner`` is True.


        Warnings
        --------
        - Using 'grid' as ``search_algorithm`` may result in very long computation.
        Only recommended with smaller search spaces that can be defined in the
        ``custom_grid`` parameter.

        - ``search_library`` 'tune-sklearn' does not support GPU models.

        """

        return super().tune_model(
            estimator=estimator,
            fold=fold,
            round=round,
            n_iter=n_iter,
            custom_grid=custom_grid,
            optimize=optimize,
            custom_scorer=custom_scorer,
            search_library=search_library,
            search_algorithm=search_algorithm,
            early_stopping=early_stopping,
            early_stopping_max_iters=early_stopping_max_iters,
            choose_better=choose_better,
            fit_kwargs=fit_kwargs,
            groups=groups,
            return_tuner=return_tuner,
            verbose=verbose,
            tuner_verbose=tuner_verbose,
            return_train_score=return_train_score,
            **kwargs,
        )

    def _plot_model(
            self,
            estimator,
            plot: str = "auc",
            scale: float = 1,  # added in pycaret_local==2.1.0
            save: Union[str, bool] = False,
            fold: Optional[Union[int, Any]] = None,
            fit_kwargs: Optional[dict] = None,
            plot_kwargs: Optional[dict] = None,
            groups: Optional[Union[str, Any]] = None,
            feature_name: Optional[str] = None,
            label: bool = False,
            use_train_data: bool = False,
            verbose: bool = True,
            system: bool = True,
            display: Optional[CommonDisplay] = None,  # added in pycaret_local==2.2.0
            display_format: Optional[str] = None,
    ) -> str:

        """Internal version of ``plot_model`` with ``system`` arg."""
        self._check_setup_ran()

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing plot_model()")
        self.logger.info(f"plot_model({function_params_str})")

        self.logger.info("Checking exceptions")

        if not fit_kwargs:
            fit_kwargs = {}

        if not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        if plot not in self._available_plots:
            raise ValueError(
                "Plot Not Available. Please see docstring for list of available Plots."
            )

        # checking display_format parameter
        self.plot_model_check_display_format_(display_format=display_format)

        # Import required libraries ----
        if display_format == "streamlit":
            _check_soft_dependencies("streamlit", extra=None, severity="error")
            import streamlit as st

        # multiclass plot exceptions:
        multiclass_not_available = ["calibration", "threshold", "manifold", "rfe"]
        if self._is_multiclass:
            if plot in multiclass_not_available:
                raise ValueError(
                    "Plot Not Available for multiclass problems. Please see docstring for list of available Plots."
                )

        # exception for CatBoost
        # if "CatBoostClassifier" in str(type(estimator)):
        #    raise ValueError(
        #    "CatBoost estimator is not compatible with plot_model function, try using Catboost with interpret_model instead."
        # )

        # checking for auc plot
        if not hasattr(estimator, "predict_proba") and plot == "auc":
            raise TypeError(
                "AUC plot not available for estimators with no predict_proba attribute."
            )

        # checking for auc plot
        if not hasattr(estimator, "predict_proba") and plot == "auc":
            raise TypeError(
                "AUC plot not available for estimators with no predict_proba attribute."
            )

        # checking for calibration plot
        if not hasattr(estimator, "predict_proba") and plot == "calibration":
            raise TypeError(
                "Calibration plot not available for estimators with no predict_proba attribute."
            )

        def is_tree(e):
            from sklearn.ensemble._forest import BaseForest
            from sklearn.tree import BaseDecisionTree

            if "final_estimator" in e.get_params():
                e = e.final_estimator
            if "base_estimator" in e.get_params():
                e = e.base_estimator
            if isinstance(e, BaseForest) or isinstance(e, BaseDecisionTree):
                return True

        # checking for calibration plot
        if plot == "tree" and not is_tree(estimator):
            raise TypeError(
                "Decision Tree plot is only available for scikit-learn Decision Trees and Forests, Ensemble models using those or Stacked models using those as meta (final) estimators."
            )

        # checking for feature plot
        if not (
                hasattr(estimator, "coef_") or hasattr(estimator, "feature_importances_")
        ) and (plot == "feature" or plot == "feature_all" or plot == "rfe"):
            raise TypeError(
                "Feature Importance and RFE plots not available for estimators that doesnt support coef_ or feature_importances_ attribute."
            )

        # checking fold parameter
        if fold is not None and not (
                type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )

        if type(label) is not bool:
            raise TypeError("Label parameter only accepts True or False.")

        if type(use_train_data) is not bool:
            raise TypeError("use_train_data parameter only accepts True or False.")

        if feature_name is not None and type(feature_name) is not str:
            raise TypeError(
                "feature parameter must be string containing column name of dataset."
            )

        """

        ERROR HANDLING ENDS HERE

        """

        cv = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        if not display:
            display = CommonDisplay(verbose=verbose, html_param=self.html_param)

        plot_kwargs = plot_kwargs or {}

        self.logger.info("Preloading libraries")
        # pre-load libraries
        import matplotlib.pyplot as plt

        np.random.seed(self.seed)

        # defining estimator as model locally
        # deepcopy instead of clone so we have a fitted estimator
        if isinstance(estimator, InternalPipeline):
            estimator = estimator.steps[-1][1]
        estimator = deepcopy(estimator)
        model = estimator

        # plots used for logging (controlled through plots_log_param)
        # AUC, #Confusion Matrix and #Feature Importance

        self.logger.info("Copying training dataset")

        self.logger.info(f"Plot type: {plot}")
        plot_name = self._available_plots[plot]

        # yellowbrick workaround start
        import yellowbrick.utils.helpers
        import yellowbrick.utils.types

        # yellowbrick workaround end

        model_name = self._get_model_name(model)
        base_plot_filename = f"{plot_name}.png"
        with patch(
                "yellowbrick.utils.types.is_estimator",
                pycaret.internal.patches.yellowbrick.is_estimator,
        ):
            with patch(
                    "yellowbrick.utils.helpers.is_estimator",
                    pycaret.internal.patches.yellowbrick.is_estimator,
            ):
                _base_dpi = 100

                def pipeline():

                    from schemdraw import Drawing
                    from schemdraw.flow import Arrow, Data, RoundBox, Subroutine

                    # Create schematic drawing
                    d = Drawing(backend="matplotlib")
                    d.config(fontsize=plot_kwargs.get("fontsize", 14))
                    d += Subroutine(w=10, h=5, s=1).label("Raw data").drop("E")
                    for est in self.pipeline:
                        name = getattr(est, "transformer", est).__class__.__name__
                        d += Arrow().right()
                        d += RoundBox(w=max(len(name), 7), h=5, cornerradius=1).label(
                            name
                        )

                    # Add the model box
                    name = estimator.__class__.__name__
                    d += Arrow().right()
                    d += Data(w=max(len(name), 7), h=5).label(name)

                    display.clear_output()

                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig, ax = plt.subplots(
                            figsize=((2 + len(self.pipeline) * 5), 6)
                        )

                        d.draw(ax=ax, showframe=False, show=False)
                        ax.set_aspect("equal")
                        plt.axis("off")
                        plt.tight_layout()

                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")

                def residuals_interactive():
                    from pycaret.internal.plots.residual_plots import (
                        InteractiveResidualsPlot,
                    )

                    resplots = InteractiveResidualsPlot(
                        x=self.X_train_transformed,
                        y=self.y_train_transformed,
                        x_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        model=estimator,
                    )

                    # display.clear_output()
                    if system:
                        resplots.show()

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        resplots.write_html(plot_filename)

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def cluster():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        estimator, verbose=False, transformation=True
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )
                    cluster = b["Cluster"].values
                    b.drop("Cluster", axis=1, inplace=True)
                    b = pd.get_dummies(b)  # casting categorical variable

                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=2, random_state=self.seed)
                    self.logger.info("Fitting PCA()")
                    pca_ = pca.fit_transform(b)
                    pca_ = pd.DataFrame(pca_)
                    pca_ = pca_.rename(columns={0: "PCA1", 1: "PCA2"})
                    pca_["Cluster"] = cluster

                    if feature_name is not None:
                        pca_["Feature"] = self.data[feature_name]
                    else:
                        pca_["Feature"] = self.data[self.data.columns[0]]

                    if label:
                        pca_["Label"] = pca_["Feature"]

                    """
                    sorting
                    """

                    self.logger.info("Sorting dataframe")

                    clus_num = [int(i.split()[1]) for i in pca_["Cluster"]]

                    pca_["cnum"] = clus_num
                    pca_.sort_values(by="cnum", inplace=True)

                    """
                    sorting ends
                    """

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    if label:
                        fig = px.scatter(
                            pca_,
                            x="PCA1",
                            y="PCA2",
                            text="Label",
                            color="Cluster",
                            opacity=0.5,
                        )
                    else:
                        fig = px.scatter(
                            pca_,
                            x="PCA1",
                            y="PCA2",
                            hover_data=["Feature"],
                            color="Cluster",
                            opacity=0.5,
                        )

                    fig.update_traces(textposition="top center")
                    fig.update_layout(plot_bgcolor="rgb(240,240,240)")

                    fig.update_layout(
                        height=600 * scale, title_text="2D Cluster PCA Plot"
                    )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

                    elif system:
                        if display_format == "streamlit":
                            st.write(fig)
                        else:
                            fig.show()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def umap():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        model, verbose=False, transformation=True, score=False
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )

                    label = pd.DataFrame(b["Anomaly"])
                    b.dropna(axis=0, inplace=True)  # droping rows with NA's
                    b.drop(["Anomaly"], axis=1, inplace=True)

                    _check_soft_dependencies(
                        "umap",
                        extra="analysis",
                        severity="error",
                        install_name="umap-learn",
                    )
                    import umap

                    reducer = umap.UMAP()
                    self.logger.info("Fitting UMAP()")
                    embedding = reducer.fit_transform(b)
                    X = pd.DataFrame(embedding)

                    import plotly.express as px

                    df = X
                    df["Anomaly"] = label

                    if feature_name is not None:
                        df["Feature"] = self.data[feature_name]
                    else:
                        df["Feature"] = self.data[self.data.columns[0]]

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    fig = px.scatter(
                        df,
                        x=0,
                        y=1,
                        color="Anomaly",
                        title="uMAP Plot for Outliers",
                        hover_data=["Feature"],
                        opacity=0.7,
                        width=900 * scale,
                        height=800 * scale,
                    )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

                    elif system:
                        if display_format == "streamlit":
                            st.write(fig)
                        else:
                            fig.show()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def tsne():
                    if self._ml_usecase == MLUsecase.CLUSTERING:
                        return _tsne_clustering()
                    else:
                        return _tsne_anomaly()

                def _tsne_anomaly():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        model, verbose=False, transformation=True, score=False
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )
                    cluster = b["Anomaly"].values
                    b.dropna(axis=0, inplace=True)  # droping rows with NA's
                    b.drop("Anomaly", axis=1, inplace=True)

                    self.logger.info("Getting dummies to cast categorical variables")

                    from sklearn.manifold import TSNE

                    self.logger.info("Fitting TSNE()")
                    X_embedded = TSNE(n_components=3).fit_transform(b)

                    X = pd.DataFrame(X_embedded)
                    X["Anomaly"] = cluster
                    if feature_name is not None:
                        X["Feature"] = self.data[feature_name]
                    else:
                        X["Feature"] = self.data[self.data.columns[0]]

                    df = X

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    if label:
                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            text="Feature",
                            color="Anomaly",
                            title="3d TSNE Plot for Outliers",
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )
                    else:
                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            hover_data=["Feature"],
                            color="Anomaly",
                            title="3d TSNE Plot for Outliers",
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

                    elif system:
                        if display_format == "streamlit":
                            st.write(fig)
                        else:
                            fig.show()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def _tsne_clustering():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        estimator,
                        verbose=False,
                        score=False,
                        transformation=True,
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )

                    cluster = b["Cluster"].values
                    b.drop("Cluster", axis=1, inplace=True)

                    from sklearn.manifold import TSNE

                    self.logger.info("Fitting TSNE()")
                    X_embedded = TSNE(
                        n_components=3, random_state=self.seed
                    ).fit_transform(b)
                    X_embedded = pd.DataFrame(X_embedded)
                    X_embedded["Cluster"] = cluster

                    if feature_name is not None:
                        X_embedded["Feature"] = self.data[feature_name]
                    else:
                        X_embedded["Feature"] = self.data[self.data.columns[0]]

                    if label:
                        X_embedded["Label"] = X_embedded["Feature"]

                    """
                    sorting
                    """
                    self.logger.info("Sorting dataframe")

                    clus_num = [int(i.split()[1]) for i in X_embedded["Cluster"]]

                    X_embedded["cnum"] = clus_num
                    X_embedded.sort_values(by="cnum", inplace=True)

                    """
                    sorting ends
                    """

                    df = X_embedded

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    if label:

                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            color="Cluster",
                            title="3d TSNE Plot for Clusters",
                            text="Label",
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )

                    else:
                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            color="Cluster",
                            title="3d TSNE Plot for Clusters",
                            hover_data=["Feature"],
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

                    elif system:
                        if display_format == "streamlit":
                            st.write(fig)
                        else:
                            fig.show()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def distribution():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    d = self.assign_model(  # type: ignore
                        estimator, verbose=False
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )

                    """
                    sorting
                    """
                    self.logger.info("Sorting dataframe")

                    clus_num = []
                    for i in d.Cluster:
                        a = int(i.split()[1])
                        clus_num.append(a)

                    d["cnum"] = clus_num
                    d.sort_values(by="cnum", inplace=True)
                    d.reset_index(inplace=True, drop=True)

                    clus_label = []
                    for i in d.cnum:
                        a = "Cluster " + str(i)
                        clus_label.append(a)

                    d.drop(["Cluster", "cnum"], inplace=True, axis=1)
                    d["Cluster"] = clus_label

                    """
                    sorting ends
                    """

                    if feature_name is None:
                        x_col = "Cluster"
                    else:
                        x_col = feature_name

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    fig = px.histogram(
                        d,
                        x=x_col,
                        color="Cluster",
                        marginal="box",
                        opacity=0.7,
                        hover_data=d.columns,
                    )

                    fig.update_layout(
                        height=600 * scale,
                    )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)

                    elif system:
                        if display_format == "streamlit":
                            st.write(fig)
                        else:
                            fig.show()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def elbow():
                    try:
                        from yellowbrick.cluster import KElbowVisualizer

                        visualizer = KElbowVisualizer(
                            estimator, timings=False, **plot_kwargs
                        )
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display_format=display_format,
                        )

                    except:
                        self.logger.error("Elbow plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def silhouette():
                    from yellowbrick.cluster import SilhouetteVisualizer

                    try:
                        visualizer = SilhouetteVisualizer(
                            estimator, colors="yellowbrick", **plot_kwargs
                        )
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display_format=display_format,
                        )
                    except:
                        self.logger.error("Silhouette plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def distance():
                    from yellowbrick.cluster import InterclusterDistance

                    try:
                        visualizer = InterclusterDistance(estimator, **plot_kwargs)
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            groups=groups,
                            display_format=display_format,
                        )
                    except:
                        self.logger.error("Distance plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def residuals():

                    from yellowbrick.regressor import ResidualsPlot

                    visualizer = ResidualsPlot(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def auc():

                    from yellowbrick.classifier import ROCAUC

                    visualizer = ROCAUC(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def threshold():

                    from yellowbrick.classifier import DiscriminationThreshold

                    visualizer = DiscriminationThreshold(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def pr():

                    from yellowbrick.classifier import PrecisionRecallCurve

                    visualizer = PrecisionRecallCurve(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def confusion_matrix():

                    from yellowbrick.classifier import ConfusionMatrix

                    plot_kwargs.setdefault("fontsize", 15)
                    plot_kwargs.setdefault("cmap", "Greens")

                    visualizer = ConfusionMatrix(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def error():

                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        from yellowbrick.classifier import ClassPredictionError

                        visualizer = ClassPredictionError(
                            estimator, random_state=self.seed, **plot_kwargs
                        )

                    elif self._ml_usecase == MLUsecase.REGRESSION:
                        from yellowbrick.regressor import PredictionError

                        visualizer = PredictionError(
                            estimator, random_state=self.seed, **plot_kwargs
                        )

                    return show_yellowbrick_plot(
                        visualizer=visualizer,  # type: ignore
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def cooks():

                    from yellowbrick.regressor import CooksDistance

                    visualizer = CooksDistance()
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        handle_test="",
                        groups=groups,
                        display_format=display_format,
                    )

                def class_report():

                    from yellowbrick.classifier import ClassificationReport

                    visualizer = ClassificationReport(
                        estimator, random_state=self.seed, support=True, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def boundary():

                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    from yellowbrick.contrib.classifier import DecisionViz

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    test_X_transformed = self.X_test_transformed.select_dtypes(
                        include="number"
                    )
                    self.logger.info("Fitting StandardScaler()")
                    data_X_transformed = StandardScaler().fit_transform(
                        data_X_transformed
                    )
                    test_X_transformed = StandardScaler().fit_transform(
                        test_X_transformed
                    )
                    pca = PCA(n_components=2, random_state=self.seed)
                    self.logger.info("Fitting PCA()")
                    data_X_transformed = pca.fit_transform(data_X_transformed)
                    test_X_transformed = pca.fit_transform(test_X_transformed)

                    viz_ = DecisionViz(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=viz_,
                        X_train=data_X_transformed,
                        y_train=np.array(self.y_train_transformed),
                        X_test=test_X_transformed,
                        y_test=np.array(self.y_test_transformed),
                        name=plot_name,
                        scale=scale,
                        handle_test="draw",
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        features=["Feature One", "Feature Two"],
                        classes=["A", "B"],
                        display_format=display_format,
                    )

                def rfe():

                    from yellowbrick.model_selection import RFECV

                    visualizer = RFECV(estimator, cv=cv, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def learning():

                    from yellowbrick.model_selection import LearningCurve

                    sizes = np.linspace(0.3, 1.0, 10)
                    visualizer = LearningCurve(
                        estimator,
                        cv=cv,
                        train_sizes=sizes,
                        n_jobs=self._gpu_n_jobs_param,
                        random_state=self.seed,
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def lift():

                    self.logger.info("Generating predictions / predict_proba on X_test")
                    y_test__ = self.y_test_transformed
                    predict_proba__ = estimator.predict_proba(self.X_test_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_lift_curve(
                            y_test__, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def gain():

                    self.logger.info("Generating predictions / predict_proba on X_test")
                    y_test__ = self.y_test_transformed
                    predict_proba__ = estimator.predict_proba(self.X_test_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_cumulative_gain(
                            y_test__, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def manifold():

                    from yellowbrick.features import Manifold

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    visualizer = Manifold(
                        manifold="tsne", random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=data_X_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit_transform",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def tree():

                    from sklearn.tree import plot_tree

                    is_stacked_model = False
                    is_ensemble_of_forests = False

                    if isinstance(estimator, InternalPipeline):
                        fitted_estimator = estimator._final_estimator
                    else:
                        fitted_estimator = estimator

                    if "final_estimator" in fitted_estimator.get_params():
                        tree_estimator = fitted_estimator.final_estimator
                        is_stacked_model = True
                    else:
                        tree_estimator = fitted_estimator

                    if (
                            "base_estimator" in tree_estimator.get_params()
                            and "n_estimators" in tree_estimator.base_estimator.get_params()
                    ):
                        n_estimators = (
                                tree_estimator.get_params()["n_estimators"]
                                * tree_estimator.base_estimator.get_params()["n_estimators"]
                        )
                        is_ensemble_of_forests = True
                    elif "n_estimators" in tree_estimator.get_params():
                        n_estimators = tree_estimator.get_params()["n_estimators"]
                    else:
                        n_estimators = 1
                    if n_estimators > 10:
                        rows = (n_estimators // 10) + 1
                        cols = 10
                    else:
                        rows = 1
                        cols = n_estimators
                    figsize = (cols * 20, rows * 16)
                    fig, axes = plt.subplots(
                        nrows=rows,
                        ncols=cols,
                        figsize=figsize,
                        dpi=_base_dpi * scale,
                        squeeze=False,
                    )
                    axes = list(axes.flatten())

                    fig.suptitle("Decision Trees")

                    self.logger.info("Plotting decision trees")
                    trees = []
                    feature_names = list(self.X_train_transformed.columns)
                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        class_names = {
                            i: class_name
                            for i, class_name in enumerate(
                                get_label_encoder(self.pipeline).classes_
                            )
                        }
                    else:
                        class_names = None
                    fitted_estimator = tree_estimator
                    if is_stacked_model:
                        stacked_feature_names = []
                        if self._ml_usecase == MLUsecase.CLASSIFICATION:
                            classes = list(self.y_train_transformed.unique())
                            if len(classes) == 2:
                                classes.pop()
                            for c in classes:
                                stacked_feature_names.extend(
                                    [
                                        f"{k}_{class_names[c]}"
                                        for k, v in fitted_estimator.estimators
                                    ]
                                )
                        else:
                            stacked_feature_names.extend(
                                [f"{k}" for k, v in fitted_estimator.estimators]
                            )
                        if not fitted_estimator.passthrough:
                            feature_names = stacked_feature_names
                        else:
                            feature_names = stacked_feature_names + feature_names
                        fitted_estimator = fitted_estimator.final_estimator_
                    if is_ensemble_of_forests:
                        for tree_estimator in fitted_estimator.estimators_:
                            trees.extend(tree_estimator.estimators_)
                    else:
                        try:
                            trees = fitted_estimator.estimators_
                        except Exception:
                            trees = [fitted_estimator]
                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        class_names = list(class_names.values())
                    for i, tree in enumerate(trees):
                        self.logger.info(f"Plotting tree {i}")
                        plot_tree(
                            tree,
                            feature_names=feature_names,
                            class_names=class_names,
                            filled=True,
                            rounded=True,
                            precision=4,
                            ax=axes[i],
                        )
                        axes[i].set_title(f"Tree {i}")
                    for i in range(len(trees), len(axes)):
                        axes[i].set_visible(False)

                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def calibration():

                    from sklearn.calibration import calibration_curve

                    plt.figure(figsize=(7, 6), dpi=_base_dpi * scale)
                    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

                    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                    self.logger.info("Scoring test/hold-out set")
                    prob_pos = estimator.predict_proba(self.X_test_transformed)[:, 1]
                    prob_pos = (prob_pos - prob_pos.min()) / (
                            prob_pos.max() - prob_pos.min()
                    )
                    (
                        fraction_of_positives,
                        mean_predicted_value,
                    ) = calibration_curve(self.y_test_transformed, prob_pos, n_bins=10)
                    ax1.plot(
                        mean_predicted_value,
                        fraction_of_positives,
                        "s-",
                        label=f"{model_name}",
                    )

                    ax1.set_ylabel("Fraction of positives")
                    ax1.set_ylim([0, 1])
                    ax1.set_xlim([0, 1])
                    ax1.legend(loc="lower right")
                    ax1.set_title("Calibration plots (reliability curve)")
                    ax1.set_facecolor("white")
                    ax1.grid(b=True, color="grey", linewidth=0.5, linestyle="-")
                    plt.tight_layout()
                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def vc():

                    self.logger.info("Determining param_name")

                    try:
                        try:
                            # catboost special case
                            model_params = estimator.get_all_params()
                        except:
                            model_params = estimator.get_params()
                    except:
                        # display.clear_output()
                        self.logger.error("VC plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError(
                            "Plot not supported for this estimator. Try different estimator."
                        )

                    param_name = ""
                    param_range = None

                    if self._ml_usecase == MLUsecase.CLASSIFICATION:

                        # Catboost
                        if "depth" in model_params:
                            param_name = "depth"
                            param_range = np.arange(1, 8 if self.gpu_param else 11)

                        # SGD Classifier
                        elif "l1_ratio" in model_params:
                            param_name = "l1_ratio"
                            param_range = np.arange(0, 1, 0.01)

                        # tree based models
                        elif "max_depth" in model_params:
                            param_name = "max_depth"
                            param_range = np.arange(1, 11)

                        # knn
                        elif "n_neighbors" in model_params:
                            param_name = "n_neighbors"
                            param_range = np.arange(1, 11)

                        # MLP / Ridge
                        elif "alpha" in model_params:
                            param_name = "alpha"
                            param_range = np.arange(0, 1, 0.1)

                        # Logistic Regression
                        elif "C" in model_params:
                            param_name = "C"
                            param_range = np.arange(1, 11)

                        # Bagging / Boosting
                        elif "n_estimators" in model_params:
                            param_name = "n_estimators"
                            param_range = np.arange(1, 1000, 10)

                        # Naive Bayes
                        elif "var_smoothing" in model_params:
                            param_name = "var_smoothing"
                            param_range = np.arange(0.1, 1, 0.01)

                        # QDA
                        elif "reg_param" in model_params:
                            param_name = "reg_param"
                            param_range = np.arange(0, 1, 0.1)

                        # GPC
                        elif "max_iter_predict" in model_params:
                            param_name = "max_iter_predict"
                            param_range = np.arange(100, 1000, 100)

                        else:
                            # display.clear_output()
                            raise TypeError(
                                "Plot not supported for this estimator. Try different estimator."
                            )

                    elif self._ml_usecase == MLUsecase.REGRESSION:

                        # Catboost
                        if "depth" in model_params:
                            param_name = "depth"
                            param_range = np.arange(1, 8 if self.gpu_param else 11)

                        # lasso/ridge/en/llar/huber/kr/mlp/br/ard
                        elif "alpha" in model_params:
                            param_name = "alpha"
                            param_range = np.arange(0, 1, 0.1)

                        elif "alpha_1" in model_params:
                            param_name = "alpha_1"
                            param_range = np.arange(0, 1, 0.1)

                        # par/svm
                        elif "C" in model_params:
                            param_name = "C"
                            param_range = np.arange(1, 11)

                        # tree based models (dt/rf/et)
                        elif "max_depth" in model_params:
                            param_name = "max_depth"
                            param_range = np.arange(1, 11)

                        # knn
                        elif "n_neighbors" in model_params:
                            param_name = "n_neighbors"
                            param_range = np.arange(1, 11)

                        # Bagging / Boosting (ada/gbr)
                        elif "n_estimators" in model_params:
                            param_name = "n_estimators"
                            param_range = np.arange(1, 1000, 10)

                        # Bagging / Boosting (ada/gbr)
                        elif "n_nonzero_coefs" in model_params:
                            param_name = "n_nonzero_coefs"
                            if len(self.X_train_transformed.columns) >= 10:
                                param_max = 11
                            else:
                                param_max = len(self.X_train_transformed.columns) + 1
                            param_range = np.arange(1, param_max, 1)

                        elif "eps" in model_params:
                            param_name = "eps"
                            param_range = np.arange(0, 1, 0.1)

                        elif "max_subpopulation" in model_params:
                            param_name = "max_subpopulation"
                            param_range = np.arange(1000, 100000, 2000)

                        elif "min_samples" in model_params:
                            param_name = "min_samples"
                            param_range = np.arange(0.01, 1, 0.1)

                        else:
                            # display.clear_output()
                            raise TypeError(
                                "Plot not supported for this estimator. Try different estimator."
                            )

                    self.logger.info(f"param_name: {param_name}")

                    from yellowbrick.model_selection import ValidationCurve

                    viz = ValidationCurve(
                        estimator,
                        param_name=param_name,
                        param_range=param_range,
                        cv=cv,
                        random_state=self.seed,
                        n_jobs=self._gpu_n_jobs_param,
                    )
                    return show_yellowbrick_plot(
                        visualizer=viz,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def dimension():

                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    from yellowbrick.features import RadViz

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    self.logger.info("Fitting StandardScaler()")
                    data_X_transformed = StandardScaler().fit_transform(
                        data_X_transformed
                    )

                    features = min(
                        round(len(self.X_train_transformed.columns) * 0.3, 0), 5
                    )
                    features = int(features)

                    pca = PCA(n_components=features, random_state=self.seed)
                    self.logger.info("Fitting PCA()")
                    data_X_transformed = pca.fit_transform(data_X_transformed)
                    classes = self.y_train_transformed.unique().tolist()
                    visualizer = RadViz(classes=classes, alpha=0.25, **plot_kwargs)

                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=data_X_transformed,
                        y_train=np.array(self.y_train_transformed),
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit_transform",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        display_format=display_format,
                    )

                def feature():
                    return _feature(10)

                def feature_all():
                    return _feature(len(self.X_train_transformed.columns))

                def _feature(n: int):
                    variables = None
                    temp_model = estimator
                    if hasattr(estimator, "steps"):
                        temp_model = estimator.steps[-1][1]
                    if hasattr(temp_model, "coef_"):
                        try:
                            coef = temp_model.coef_.flatten()
                            if len(coef) > len(self.X_train_transformed.columns):
                                coef = coef[: len(self.X_train_transformed.columns)]
                            variables = abs(coef)
                        except:
                            pass
                    if variables is None:
                        self.logger.warning(
                            "No coef_ found. Trying feature_importances_"
                        )
                        variables = abs(temp_model.feature_importances_)
                    coef_df = pd.DataFrame(
                        {
                            "Variable": self.X_train_transformed.columns,
                            "Value": variables,
                        }
                    )
                    sorted_df = (
                        coef_df.sort_values(by="Value", ascending=False)
                            .head(n)
                            .sort_values(by="Value")
                    )
                    my_range = range(1, len(sorted_df.index) + 1)
                    plt.figure(figsize=(8, 5 * (n // 10)), dpi=_base_dpi * scale)
                    plt.hlines(
                        y=my_range,
                        xmin=0,
                        xmax=sorted_df["Value"],
                        color="skyblue",
                    )
                    plt.plot(sorted_df["Value"], my_range, "o")
                    plt.yticks(my_range, sorted_df["Variable"])
                    plt.title("Feature Importance Plot")
                    plt.xlabel("Variable Importance")
                    plt.ylabel("Features")
                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def parameter():

                    try:
                        params = estimator.get_all_params()
                    except:
                        params = estimator.get_params(deep=False)

                    param_df = pd.DataFrame.from_dict(
                        {str(k): str(v) for k, v in params.items()},
                        orient="index",
                        columns=["Parameters"],
                    )
                    # use ipython directly to show it in the widget
                    ipython_display(param_df)
                    self.logger.info("Visual Rendered Successfully")

                def ks():

                    self.logger.info("Generating predictions / predict_proba on X_test")
                    predict_proba__ = estimator.predict_proba(self.X_train_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig = skplt.metrics.plot_ks_statistic(
                            self.y_train_transformed, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def plot_cauc():
                    temp_model = estimator
                    max_time_train = np.floor(self.survival_train["time"].max()-0.5)
                    min_time_train = np.ceil(self.survival_train["time"].min()+0.5)
                    x_test = self.X_test_transformed
                    survival_test = Surv.from_arrays(event=self.y_test_transformed, time=x_test["time"])
                    x_test = self.X_test_transformed[(self.X_test_transformed['time'] > min_time_train) & (self.X_test_transformed['time'] < max_time_train)]

                    survival_test = survival_test[(survival_test["time"] < max_time_train) & (survival_test["time"] > min_time_train)]
                    max_time_test = survival_test["time"].max()
                    min_time_test = survival_test["time"].min()
                    times = np.linspace(np.ceil(min_time_test+0.01), np.floor(max_time_test-0.1), 50)

                    if hasattr(estimator, "_predict_cumulative_hazard_function"):
                        risk_scores = np.row_stack([fn(times) for fn in temp_model._predict_cumulative_hazard_function(x_test)])
                    elif hasattr(estimator, "predict_cumulative_hazard_function"):
                        risk_scores = np.row_stack([fn(times) for fn in temp_model.predict_cumulative_hazard_function(x_test)])
                    else:
                        self.logger.warning("The model does not have a predict_cumulative_hazard_function")
                        return None
                    surv_auc, mean_auc = cumulative_dynamic_auc(self.survival_train, survival_test, risk_scores, times)
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig, ax = plt.subplots(figsize=(9, 6))
                        ax.plot(times, surv_auc, marker="o", label=temp_model.__class__.__bases__[
                                                                        -1].__name__ + " Cumulative Hazard AUC")
                        ax.axhline(mean_auc, label="Mean AUC", color='red', linestyle="--")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Time-dependent Cumulative Hazard AUC")
                        ax.set_title("Cumulative Hazard AUC")
                        ax.grid(True)
                        ax.legend()
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def plot_nzcoefs():
                    temp_model = estimator
                    Xt = self.X_train.loc[:, self.X_train.columns != 'time'].to_numpy(copy=True)
                    y_time = self.X_train["time"].values.ravel()
                    y = Surv.from_arrays(event=self.y_train, time=y_time)

                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", FitFailedWarning)
                    estimated_alphas = temp_model.alphas_
                    cv_temp = KFold(n_splits=5, shuffle=True, random_state=0)
                    gcv = GridSearchCV(
                        make_pipeline(StandardScaler(), temp_model.__class__.__bases__[-1]()),
                        param_grid={temp_model.__class__.__bases__[-1].__name__.lower() + "__alphas": [[v] for v in
                                                                                                       estimated_alphas]},
                        cv=cv_temp,
                        error_score=0.5,
                        n_jobs=1).fit(Xt, y)

                    best_model = gcv.best_estimator_.named_steps[temp_model.__class__.__bases__[-1].__name__.lower()]
                    best_coefs = pd.DataFrame(
                        best_model.coef_,
                        index=[column for column in self.X_train_transformed.columns if column != "time"],
                        columns=["coefficient"]
                    )
                    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)

                    print("Number of non-zero coefficients: {}".format(non_zero))

                    non_zero_coefs = best_coefs.query("coefficient != 0")
                    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

                    plot_filename = None
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig, ax = plt.subplots(figsize=(9, 6))
                        non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
                        ax.set_xlabel("coefficient")
                        ax.grid(True)
                        ax.set_title("Non-zero coefficients")
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename


                def plot_grouped_survival():
                    return plot_filename

                def plot_survival_curve():
                    temp_model = estimator
                    x_test = self.X_test_transformed
                    survival_test = Surv.from_arrays(event=self.y_test_transformed, time=x_test["time"])
                    from sksurv.nonparametric import kaplan_meier_estimator
                    survival_pred = temp_model.predict(self.X_test_transformed)
                    str_pred = ['Low', 'High']
                    id_pred = [0, 1]
                    group_prediction = [id_pred[1] if pred > np.median(survival_pred) else id_pred[0] for pred in survival_pred]

                    surv_test_with_group = np.empty(len(survival_test), dtype=[('event', bool), ('time', float), ('group', int)])
                    a = np.asanyarray(group_prediction)

                    surv_test_with_group['group'] = np.array(group_prediction)
                    surv_test_with_group['event'] = survival_test['event']
                    surv_test_with_group['time'] = survival_test['time']
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig, ax = plt.subplots(figsize=(9, 6))
                        for group in np.unique(surv_test_with_group['group']):
                            time_cell, survival_prob_cell = kaplan_meier_estimator(
                                surv_test_with_group[surv_test_with_group['group'] == group]['event'],
                                surv_test_with_group[surv_test_with_group['group'] == group]['time'])
                            ax.step(time_cell,
                                    survival_prob_cell,
                                    where="post",
                                    label="%s (n = %d)" % (str_pred[group], np.sum(surv_test_with_group['group'] == group)))
                        ax.set_ylabel("est. probability of survival $\hat{S}(t)$")
                        ax.set_xlabel("time $t$")
                        ax.legend(loc="best")
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()
                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename


                def plot_cindex():
                    temp_model = estimator
                    Xt = self.X_train.loc[:, self.X_train.columns != 'time'].to_numpy(copy=True)
                    y_time = self.X_train["time"].values.ravel()
                    y = Surv.from_arrays(event=self.y_train, time=y_time)

                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", FitFailedWarning)
                    estimated_alphas = temp_model.alphas_
                    cv_temp = KFold(n_splits=5, shuffle=True, random_state=0)
                    gcv = GridSearchCV(
                        make_pipeline(StandardScaler(), temp_model.__class__.__bases__[-1]()),
                        param_grid={temp_model.__class__.__bases__[-1].__name__.lower() + "__alphas": [[v] for v in
                                                                                                       estimated_alphas]},
                        cv=cv_temp,
                        error_score=0.5,
                        n_jobs=1).fit(Xt, y)
                    cv_results = pd.DataFrame(gcv.cv_results_)
                    alphas = pd.Series(gcv.cv_results_['param_' + temp_model.__class__.__bases__[
                        -1].__name__.lower() + "__alphas"]).map(lambda x: x[0])
                    mean = cv_results.mean_test_score
                    std = cv_results.std_test_score
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig, ax = plt.subplots(figsize=(9, 6))
                        ax.plot(alphas, mean)
                        ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
                        ax.set_xscale("log")
                        ax.set_ylabel("concordance index")
                        ax.set_xlabel("alpha")
                        ax.axvline(
                            gcv.best_params_[temp_model.__class__.__bases__[-1].__name__.lower() + "__alphas"][0],
                            c="C1")
                        ax.axhline(0.5, color="grey", linestyle="--")
                        ax.grid(True)
                        # plt.title("C-Index Plot")
                        # plt.xlabel("alpha")
                        # plt.ylabel("coefficient")
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def plot_coefficients(**kwargs):
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        temp_model = estimator
                        coefs = pd.DataFrame(
                            temp_model.coef_,
                            index=[column for column in self.X_train_transformed.columns if column != "time"],
                            columns=np.round(temp_model.alphas_, 5)
                        )
                        _, ax = plt.subplots(figsize=(8, 5), dpi=_base_dpi * scale)
                        n_features = coefs.shape[0]
                        alphas = coefs.columns
                        for row in coefs.itertuples():
                            ax.semilogx(alphas, row[1:], ".-", label=row.Index)

                        alpha_min = alphas.min()
                        top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(5)
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

                        plt.title("Coefficients")
                        plt.xlabel("alpha")
                        plt.ylabel("coefficient")
                        plot_filename = 'Coefficients_survival_plot.png'
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def plot_coefficients(**kwargs):
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        temp_model = estimator
                        coefs = pd.DataFrame(
                            temp_model.coef_,
                            index=[column for column in self.X_train_transformed.columns if column != "time"],
                            columns=np.round(temp_model.alphas_, 5)
                        )
                        _, ax = plt.subplots(figsize=(8, 5), dpi=_base_dpi * scale)
                        n_features = coefs.shape[0]
                        alphas = coefs.columns
                        for row in coefs.itertuples():
                            ax.semilogx(alphas, row[1:], ".-", label=row.Index)

                        alpha_min = alphas.min()
                        top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(5)
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

                        plt.title("Coefficients")
                        plt.xlabel("alpha")
                        plt.ylabel("coefficient")
                        plot_filename = 'Coefficients_survival_plot.png'
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def coefficients_surv():
                    ax = estimator.lifelines_model.plot()
                    plt.title("Survival Analysis Coefficients")
                    plt.xlabel("log(HR) 95% CI")
                    plt.ylabel("Features")
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename



                def partial_effects_on_outcome():
                    if "covariates" in plot_kwargs:
                        covariates = plot_kwargs.pop("covariates")
                    # else:
                    #     raise ValueError("covariates must be specified.")
                    if "values" in plot_kwargs:
                        values = plot_kwargs.pop("values")
                    # else:
                    #     raise ValueError("values must be specified.")
                    return _partial_effects_on_outcome(covariates=covariates, values=values, **plot_kwargs)

                def _partial_effects_on_outcome(covariates: Any, values: Any):
                    temp = estimator.lifelines_model.plot_partial_effects_on_outcome(covariates=covariates,
                                                                                   values=values,
                                                                                   **plot_kwargs)
                    plt.title("Partial Effects on Outcome")
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def test_plot():

                    # from yellowbrick.regressor import ResidualsPlot
                    # estimator.lifelines_model.plot()
                    # visualizer = ResidualsPlot(estimator, **plot_kwargs)
                    # plt = estimator.lifelines_model.plot()
                    # plt.figure(estimator.lifelines_model.plot())
                    # plt.show()
                    lifeline_model = estimator.lifelines_model
                    axes_sublot = lifeline_model.plot()
                    return "estimator.lifelines_model.plot().show()"

                # def

                # execute the plot method
                with redirect_output(self.logger):
                    ret = locals()[plot]()
                if ret:
                    plot_filename = ret
                else:
                    plot_filename = base_plot_filename

                try:
                    plt.close()
                except:
                    pass

        gc.collect()

        self.logger.info(
            "plot_model() successfully completed......................................"
        )

        if save:
            return plot_filename

    def plot_model(
            self,
            estimator,
            plot: str = "auc",
            scale: float = 1,  # added in pycaret_local==2.1.0
            save: Union[str, bool] = False,
            fold: Optional[Union[int, Any]] = None,
            fit_kwargs: Optional[dict] = None,
            plot_kwargs: Optional[dict] = None,
            groups: Optional[Union[str, Any]] = None,
            feature_name: Optional[str] = None,
            label: bool = False,
            use_train_data: bool = False,
            verbose: bool = True,
            display_format: Optional[str] = None,
    ) -> Optional[str]:

        """
        This function takes a trained model object and returns a plot based on the
        test / hold-out set. The process may require the model to be re-trained in
        certain cases. See list of plots supported below.

        Model must be created using create_model() or tune_model().

        Example
        -------
        >>> from pycaret_local.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> plot_model(lr)

        This will return an AUC plot of a trained Logistic Regression model.

        Parameters
        ----------
        estimator : object, default = none
            A trained model object should be passed as an estimator.

        plot : str, default = auc
            Enter abbreviation of type of plot. The current list of plots supported are (Plot - Name):

            * 'pipeline' - Schematic drawing of the preprocessing pipeline
            * 'residuals_interactive' - Interactive Residual plots
            * 'auc' - Area Under the Curve
            * 'threshold' - Discrimination Threshold
            * 'pr' - Precision Recall Curve
            * 'confusion_matrix' - Confusion Matrix
            * 'error' - Class Prediction Error
            * 'class_report' - Classification Report
            * 'boundary' - Decision Boundary
            * 'rfe' - Recursive Feature Selection
            * 'learning' - Learning Curve
            * 'manifold' - Manifold Learning
            * 'calibration' - Calibration Curve
            * 'vc' - Validation Curve
            * 'dimension' - Dimension Learning
            * 'feature' - Feature Importance
            * 'feature_all' - Feature Importance (All)
            * 'parameter' - Model Hyperparameter
            * 'lift' - Lift Curve
            * 'gain' - Gain Chart

        scale: float, default = 1
            The resolution scale of the figure.

        save: string or bool, default = False
            When set to True, Plot is saved as a 'png' file in current working directory.
            When a path destination is given, Plot is saved as a 'png' file the given path to the directory of choice.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation used in certain plots. If None, will use the CV generator
            defined in setup(). If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        verbose: bool, default = True
            Progress bar not shown when verbose set to False.

        system: bool, default = True
            Must remain True all times. Only to be changed by internal functions.

        display_format: str, default = None
            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.
            Currently, not all plots are supported.

        Returns
        -------
        Visual_Plot
            Prints the visual plot.
        str:
            If save parameter is True, will return the name of the saved file.

        Warnings
        --------
        -  'svm' and 'ridge' doesn't support the predict_proba method. As such, AUC and
            calibration plots are not available for these estimators.

        -   When the 'max_features' parameter of a trained model object is not equal to
            the number of samples in training set, the 'rfe' plot is not available.

        -   'calibration', 'threshold', 'manifold' and 'rfe' plots are not available for
            multiclass problems.


        """
        return self._plot_model(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
            groups=groups,
            feature_name=feature_name,
            label=label,
            use_train_data=use_train_data,
            verbose=verbose,
            display_format=display_format,
        )

    def evaluate_model(
            self,
            estimator,
            fold: Optional[Union[int, Any]] = None,
            fit_kwargs: Optional[dict] = None,
            plot_kwargs: Optional[dict] = None,
            feature_name: Optional[str] = None,
            groups: Optional[Union[str, Any]] = None,
            use_train_data: bool = False,
    ):

        """
        This function displays a user interface for all of the available plots for
        a given estimator. It internally uses the plot_model() function.

        Example
        -------
        >>> from pycaret_local.datasets import get_data
        >>> juice = get_data('juice')
        >>> experiment_name = setup(data = juice,  target = 'Purchase')
        >>> lr = create_model('lr')
        >>> evaluate_model(lr)

        This will display the User Interface for all of the plots for a given
        estimator.

        Parameters
        ----------
        estimator : object, default = none
            A trained model object should be passed as an estimator.

        fold: integer or scikit-learn compatible CV generator, default = None
            Controls cross-validation. If None, will use the CV generator defined in setup().
            If integer, will use KFold CV with that many folds.
            When cross_validation is False, this parameter is ignored.

        fit_kwargs: dict, default = {} (empty dict)
            Dictionary of arguments passed to the fit method of the model.

        groups: str or array-like, with shape (n_samples,), default = None
            Optional Group labels for the samples used while splitting the dataset into train/test set.
            If string is passed, will use the data column with that name as the groups.
            Only used if a group based cross-validation generator is used (eg. GroupKFold).
            If None, will use the value set in fold_groups parameter in setup().

        Returns
        -------
        User_Interface
            Displays the user interface for plotting.

        """

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing evaluate_model()")
        self.logger.info(f"evaluate_model({function_params_str})")

        from ipywidgets import widgets
        from ipywidgets.widgets import fixed, interact

        if not fit_kwargs:
            fit_kwargs = {}

        a = widgets.ToggleButtons(
            options=[(v, k) for k, v in self._available_plots.items()],
            description="Plot Type:",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            icons=[""],
        )

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        interact(
            self._plot_model,
            estimator=fixed(estimator),
            plot=a,
            save=fixed(False),
            verbose=fixed(False),
            scale=fixed(1),
            fold=fixed(fold),
            fit_kwargs=fixed(fit_kwargs),
            plot_kwargs=fixed(plot_kwargs),
            feature_name=fixed(feature_name),
            label=fixed(False),
            groups=fixed(groups),
            use_train_data=fixed(use_train_data),
            system=fixed(True),
            display=fixed(None),
            display_format=fixed(None),
        )