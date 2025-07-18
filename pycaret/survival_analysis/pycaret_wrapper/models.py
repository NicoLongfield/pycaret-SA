# Module: containers.models.regression
# Author: Moez Ali <moez.ali@queensu.ca> and Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of regression models. The `regression` module will
# call `get_all_model_containers()`, which will return instances of all classes in this module that have `RegressionContainer`
# as a base (but not `RegressionContainer` itself). In order to add a new model, you only need to create a new class that has
# `RegressionContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

import logging
from typing import Any, Dict, Optional, Union, List

import numpy as np
import pandas as pd
from packaging import version

import pycaret.containers.base_container
from pycaret.containers.models.base_model import (
    ModelContainer,
    leftover_parameters_to_categorical_distributions,
)
from pycaret.internal.distributions import (
    Distribution,
    IntUniformDistribution,
    UniformDistribution,
)
from pycaret.utils.generic import get_logger, np_list_arange, param_grid_to_lists
from pycaret.utils._dependencies import _check_soft_dependencies
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, IPCRidge
from sksurv.tree import SurvivalTree
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.svm import (HingeLossSurvivalSVM,
                        FastSurvivalSVM,
                        FastKernelSurvivalSVM,
                        MinlipSurvivalAnalysis,
                        NaiveSurvivalSVM)

from sksurv.ensemble import (RandomSurvivalForest,
                             ComponentwiseGradientBoostingSurvivalAnalysis,
                             ExtraSurvivalTrees,
                             GradientBoostingSurvivalAnalysis,)



class SurvivalAnalysisContainer(ModelContainer):
    """
    Base regression model container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool, default = True
        Should the model be used with 'turbo = True' in compare_models().
    eq_function : type, default = None
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool, default = False
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list, default = {} (empty dict)
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution, default = {} (empty dict)
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict, default = {} (empty dict)
        The arguments to always pass to the tuner.
    shap : bool or str, default = False
        If False, SHAP is not supported. Otherwise, one of 'type1', 'type2' to determine SHAP type.
    is_gpu_enabled : bool, default = None
        If None, will try to automatically determine.
    is_boosting_supported : bool, default = None
        If None, will try to automatically determine.
    tunable : type, default = None
        If a special tunable model is used for tuning, type of
        that model, else None.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool
        Should the model be used with 'turbo = True' in compare_models().
    eq_function : type
        Function to use to check whether an object (model) can be considered equal to the model
        in the container. If None, will be ``is_instance(x, class_def)`` where x is the object.
    args : dict
        The arguments to always pass to constructor when initializing object of class_def class.
    is_special : bool
        Is the model special (not intended to be used on its own, eg. VotingClassifier).
    tune_grid : dict of str : list
        The hyperparameters tuning grid for random and grid search.
    tune_distribution : dict of str : Distribution
        The hyperparameters tuning grid for other types of searches.
    tune_args : dict
        The arguments to always pass to the tuner.
    shap : bool or str
        If False, SHAP is not supported. Otherwise, one of 'type1', 'type2' to determine SHAP type.
    is_gpu_enabled : bool
        If None, will try to automatically determine.
    is_boosting_supported : bool
        If None, will try to automatically determine.
    tunable : type
        If a special tunable model is used for tuning, type of
        that model, else None.

    """

    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        is_turbo: bool = True,
        eq_function: Optional[type] = None,
        args: Dict[str, Any] = None,
        is_special: bool = False,
        tune_grid: Dict[str, list] = None,
        tune_distribution: Dict[str, Distribution] = None,
        tune_args: Dict[str, Any] = None,
        shap: Union[bool, str] = False,
        is_gpu_enabled: Optional[bool] = None,
        tunable: Optional[type] = None,
    ) -> None:

        self.shap = shap
        if not (isinstance(shap, bool) or shap in ["type1", "type2"]):
            raise ValueError("shap must be either bool or 'type1', 'type2'.")

        if not args:
            args = {}

        if not tune_grid:
            tune_grid = {}

        if not tune_distribution:
            tune_distribution = {}

        if not tune_args:
            tune_args = {}

        super().__init__(
            id=id,
            name=name,
            class_def=class_def,
            eq_function=eq_function,
            args=args,
            is_special=is_special,
        )
        self.is_turbo = is_turbo
        self.tune_grid = param_grid_to_lists(tune_grid)
        self.tune_distribution = tune_distribution
        self.tune_args = tune_args
        self.tunable = tunable

        self.is_boosting_supported = True
        self.is_soft_voting_supported = True

        if is_gpu_enabled is not None:
            self.is_gpu_enabled = is_gpu_enabled
        else:
            self.is_gpu_enabled = bool(self.get_package_name() == "cuml")

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        """
        Returns a dictionary of the model properties, to
        be turned into a pandas DataFrame row.

        Parameters
        ----------
        internal : bool, default = True
            If True, will return all properties. If False, will only
            return properties intended for the user to see.

        Returns
        -------
        dict of str : Any

        """
        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("Reference", self.reference),
            ("Turbo", self.is_turbo),
        ]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Equality", self.eq_function),
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Distributions", self.tune_distribution),
                ("Tune Args", self.tune_args),
                ("SHAP", self.shap),
                ("GPU Enabled", self.is_gpu_enabled),
                ("Tunable Class", self.tunable),
            ]

        return dict(d)


class IPCRidgeWrapper(IPCRidge):
    """Accelerated failure time model with inverse probability of censoring weights.

    This model assumes a regression model of the form

    .. math::

        \\log y = \\beta_0 + \\mathbf{X} \\beta + \\epsilon

    L2-shrinkage is applied to the coefficients :math:`\\beta` and
    each sample is weighted by the inverse probability of censoring
    to account for right censoring (under the assumption that
    censoring is independent of the features, i.e., random censoring).

    See [1]_ for further description.

    Parameters
    ----------
    alpha : float, optional, default: 1.0
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.

    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Weight vector.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] W. Stute, "Consistent estimation under random censorship when covariables are
           present", Journal of Multivariate Analysis, vol. 45, no. 1, pp. 89-103, 1993.
           doi:10.1006/jmva.1993.1028.
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize="deprecated",
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto", **kwargs):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        # Set estimator type to None to avoid sklearn trying to categorize it
        self._estimator_type = None
        # Ignore any additional keyword arguments passed by PyCaret
        super(IPCRidge, self).__init__()

    def _more_tags(self):
        return {'allow_multiclass': False, 'allow_multilabel': False, 'requires_survival': True}
        
    def _get_tags(self):
        tags = super()._get_tags() if hasattr(super(), '_get_tags') else {}
        tags.update(self._more_tags())
        return tags

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)

        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv)

    def predict(self, X: pd.DataFrame, alpha: Optional[float] = None):
        return super().predict(X.loc[:, X.columns != 'time'])

    def predict_survival_function(self, X: pd.DataFrame, alpha: Optional[float] = None):
        # IPCRidge doesn't natively support survival function prediction
        # Return None to indicate this method is not available
        return None

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class IPCRidgeSurvivalAnalysisContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
            "fit_intercept": [True, False],
            "normalize": [True, False],
        }
        tune_distributions = {"alpha": UniformDistribution(0.001, 10)}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ipc_ridge",
            name="IPC Ridge",
            class_def=IPCRidgeWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class CoxPHWrapper(CoxPHSurvivalAnalysis):
    """Cox proportional hazards model.

    There are two possible choices for handling tied event times.
    The default is Breslow's method, which considers each of the
    events at a given time as distinct. Efron's method is more
    accurate if there are a large number of ties. When the number
    of ties is small, the estimated coefficients by Breslow's and
    Efron's method are quite close. Uses Newton-Raphson optimization.

    See [1]_, [2]_, [3]_ for further description.

    Parameters
    ----------
    alpha : float, ndarray of shape (n_features,), optional, default: 0
        Regularization parameter for ridge regression penalty.
        If a single float, the same penalty is used for all features.
        If an array, there must be one penalty for each feature.
        If you want to include a subset of features without penalization,
        set the corresponding entries to 0.

    ties : "breslow" | "efron", optional, default: "breslow"
        The method to handle tied event times. If there are
        no tied event times all the methods are equivalent.

    n_iter : int, optional, default: 100
        Maximum number of iterations.

    tol : float, optional, default: 1e-9
        Convergence criteria. Convergence is based on the negative log-likelihood::

        |1 - (new neg. log-likelihood / old neg. log-likelihood) | < tol

    verbose : int, optional, default: 0
        Specified the amount of additional debug information
        during optimization.

    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Coefficients of the model

    cum_baseline_hazard_ : :class:`sksurv.functions.StepFunction`
        Estimated baseline cumulative hazard function.

    baseline_survival_ : :class:`sksurv.functions.StepFunction`
        Estimated baseline survival function.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.

    See also
    --------
    sksurv.linear_model.CoxnetSurvivalAnalysis
        Cox proportional hazards model with l1 (LASSO) and l2 (ridge) penalty.

    References
    ----------
    .. [1] Cox, D. R. Regression models and life tables (with discussion).
           Journal of the Royal Statistical Society. Series B, 34, 187-220, 1972.
    .. [2] Breslow, N. E. Covariance Analysis of Censored Survival Data.
           Biometrics 30 (1974): 89–99.
    .. [3] Efron, B. The Efficiency of Cox’s Likelihood Function for Censored Data.
           Journal of the American Statistical Association 72 (1977): 557–565.
    """

    def __init__(self, alpha: float = 0.00001, ties: str = "breslow", n_iter: int = 100, tol: float = 1e-9, verbose: int = 0, **kwargs):
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        # Set estimator type to None to avoid sklearn trying to categorize it
        self._estimator_type = None

        self._baseline_model = BreslowEstimator()

        super(CoxPHSurvivalAnalysis, self).__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)

        return super().fit(X.loc[:, X.columns != 'time'], y_surv)

    def predict(self, X: pd.DataFrame, alpha: Optional[float] = None):
        return super().predict(X.loc[:, X.columns != 'time'])

    def predict_survival_function(self, X: pd.DataFrame, alpha: Optional[float] = None):
        return super().predict_survival_function(X.loc[:, X.columns != 'time'])

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class CoxPHSurvivalAnalysisContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": 10.**np.linspace(-5, 5, 10),
            "ties": ['breslow', 'efron'],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0, 1),
        }


        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="coxph",
            name="CoxPH",
            class_def=CoxPHWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class CoxNetWrapper(CoxnetSurvivalAnalysis):

    """Cox's proportional hazard's model with elastic net penalty.
    See the :ref:`User Guide </user_guide/coxnet.ipynb>` and [1]_ for further description.
    Parameters
    ----------
    n_alphas : int, optional, default: 100
        Number of alphas along the regularization path.
    alphas : array-like or None, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically.
    alpha_min_ratio : float or { "auto" }, optional, default: "auto"
        Determines minimum alpha of the regularization path
        if ``alphas`` is ``None``. The smallest value for alpha
        is computed as the fraction of the data derived maximum
        alpha (i.e. the smallest value for which all
        coefficients are zero).
        If set to "auto", the value will depend on the
        sample size relative to the number of features.
        If ``n_samples > n_features``, the default value is 0.0001
        If ``n_samples <= n_features``, 0.01 is the default value.
    l1_ratio : float, optional, default: 0.5
        The ElasticNet mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty.
        For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.
    penalty_factor : array-like or None, optional
        Separate penalty factors can be applied to each coefficient.
        This is a number that multiplies alpha to allow differential
        shrinkage.  Can be 0 for some variables, which implies no shrinkage,
        and that variable is always included in the model.
        Default is 1 for all variables. Note: the penalty factors are
        internally rescaled to sum to n_features, and the alphas sequence
        will reflect this change.
    normalize : boolean, optional, default: False
        If True, the features X will be normalized before optimization by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.
    copy_X : boolean, optional, default: True
        If ``True``, X will be copied; else, it may be overwritten.
    tol : float, optional, default: 1e-7
        The tolerance for the optimization: optimization continues
        until all updates are smaller than ``tol``.
    max_iter : int, optional, default: 100000
        The maximum number of iterations.
    verbose : bool, optional, default: False
        Whether to print additional information during optimization.
    fit_baseline_model : bool, optional, default: False
        Whether to estimate baseline survival function
        and baseline cumulative hazard function for each alpha.
        If enabled, :meth:`predict_cumulative_hazard_function` and
        :meth:`predict_survival_function` can be used to obtain
        predicted  cumulative hazard function and survival function.
    Attributes
    ----------
    alphas_ : ndarray, shape=(n_alphas,)
        The actual sequence of alpha values used.
    alpha_min_ratio_ : float
        The inferred value of alpha_min_ratio.
    penalty_factor_ : ndarray, shape=(n_features,)
        The actual penalty factors used.
    coef_ : ndarray, shape=(n_features, n_alphas)
        Matrix of coefficients.
    offset_ : ndarray, shape=(n_alphas,)
        Bias term to account for non-centered features.
    deviance_ratio_ : ndarray, shape=(n_alphas,)
        The fraction of (null) deviance explained.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    References
    ----------
    .. [1] Simon N, Friedman J, Hastie T, Tibshirani R.
           Regularization paths for Cox’s proportional hazards model via coordinate descent.
           Journal of statistical software. 2011 Mar;39(5):1.
    """
    def __init__(self, n_alphas: int = 100, alphas: Optional[List[float]] = None,
                 alpha_min_ratio: str = "auto", l1_ratio: float = 0.5,
                 penalty_factor: Optional[List[float]] = None, normalize: bool = False,
                 copy_X: bool = True, tol: float = 1e-7, max_iter: int = 100000,
                 verbose: bool = False, fit_baseline_model: bool = True, prediction_function: str = "survival", **kwargs):

        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio = l1_ratio
        self.penalty_factor = penalty_factor
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.fit_baseline_model = fit_baseline_model
        self.prediction_function = prediction_function
        # Set estimator type to None to avoid sklearn trying to categorize it
        self._estimator_type = None

        super(CoxnetSurvivalAnalysis, self).__init__()
                    # self, n_alphas=self.n_alphas, alphas=self.alphas,
                    # alpha_min_ratio=self.alpha_min_ratio, l1_ratio=self.l1_ratio,
                    # penalty_factor=self.penalty_factor, normalize=self.normalize,
                    # copy_X=self.copy_X, tol=self.tol, max_iter=self.max_iter,
                    # verbose=self.verbose, fit_baseline_model=self.fit_baseline_model)
    # def __init__(self, model: Any, **kwargs):
    #     self.model = model(**kwargs)

    def __init__(self, **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        # Remove common sklearn parameters that aren't relevant for survival models
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        # Pass the filtered kwargs to the parent class
        super().__init__(**filtered_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)

        return super().fit(X.loc[:, X.columns != 'time'], y_surv)

    def predict(self, X: pd.DataFrame, alpha: Optional[float] = None):
        return super().predict(X.loc[:, X.columns != 'time'], alpha)

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


    def predict_survival_function(self, X: pd.DataFrame, alpha: Optional[float] = None):
        if hasattr(super(), 'predict_survival_function'):
            return super().predict_survival_function(X.loc[:, X.columns != 'time'], alpha)
        else:
            # If the method doesn't exist, return None
            return None
    
    def predict_cumulative_hazard_function(self, X: pd.DataFrame, alpha: Optional[float] = None):
        if hasattr(super(), 'predict_cumulative_hazard_function'):
            return super().predict_cumulative_hazard_function(X.loc[:, X.columns != 'time'], alpha)
        else:
            # If the method doesn't exist, return None
            return None

    #
    # def _pre_fit(self, X, y):
    #     y = Surv.from_dataframe(event=y.columns[0], time=y.columns[1], data=y)
    #     return super._pre_fit(X, y)


class CoxNetSurvivalAnalysisContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        # y = Surv.from_dataframe(experiment.)
        # CoxPHRegression = ModelWrapperCoxPH()

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha_min_ratio": 10. ** np.linspace(-8, -0.001, 10),
            "l1_ratio": 10. ** np.linspace(-8, -0.001, 10),
            "normalize": [True, False],
        }
        tune_distributions = {
            "alpha_min_ratio": UniformDistribution(0, 1),
            "l1_ratio": UniformDistribution(0.01, 0.9999999999),
        }


        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="coxnet",
            name="CoxNet",
            class_def=CoxNetWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )

class CoxNetLassoSurvivalAnalysisContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False



        args = {"l1_ratio": 1.0, "alpha_min_ratio": "auto"}
        tune_args = {}
        tune_grid = {
            # "alpha_min_ratio": 10. ** np.linspace(-8, -0.001, 10),
            "normalize": [True, False],
        }
        tune_distributions = {
            # "alphas": UniformDistribution(0, 1),
        }


        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lasso-coxnet",
            name="CoxNetLasso",
            class_def=CoxNetWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )



class CoxNetElasticSurvivalAnalysisContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False


        # args = {"alpha_min_ratio": "auto"}
        args = {"n_alphas": 100, "alpha_min_ratio": "auto"}
        tune_args = {}

        tune_grid = {
            # "alpha_min_ratio": 10. ** np.linspace(-8, -0.001, 10),
            "l1_ratio": 10. ** np.linspace(-8, -0.001, 10),
            "normalize": [True, False],
        }
        tune_distributions = {
            "l1_ratio": UniformDistribution(0.01, 0.9999999999),
            # "alpha_min_ratio": UniformDistribution(0, 1),
        }


        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="elastic-coxnet",
            name="CoxNetLasso",
            class_def=CoxNetWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )

class SurvivalTreeWrapper(SurvivalTree):

    """A survival tree.
    The quality of a split is measured by the
    log-rank splitting rule.
    See [1]_, [2]_ and [3]_ for further description.
    Parameters
    ----------
    splitter : string, optional, default: "best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.
    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    Attributes
    ----------
    event_times_ : array of shape = (n_event_times,)
        Unique time points where events occurred.
    max_features_ : int,
        The inferred value of max_features.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    tree_ : Tree object
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object.
    See also
    --------
    sksurv.ensemble.RandomSurvivalForest
        An ensemble of SurvivalTrees.
    References
    ----------
    .. [1] Leblanc, M., & Crowley, J. (1993). Survival Trees by Goodness of Split.
           Journal of the American Statistical Association, 88(422), 457–467.
    .. [2] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841–860.
    .. [3] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25–31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    def __init__(self,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        # Remove common sklearn parameters that aren't relevant for survival trees
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        # super(SurvivalTree, self).__init__()
        super().__init__(splitter=splitter,
                         random_state=random_state,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv)

    def predict(self, X: pd.DataFrame):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)

    def predict_survival_function(self, X):
        return super().predict_survival_function(X.loc[:, X.columns != 'time'].to_numpy(copy=True), check_input=False)

    def _predict_cumulative_hazard_function(self, X):
        return super().predict_cumulative_hazard_function(X.loc[:, X.columns != 'time'].to_numpy(copy=True),
                                                          check_input=False,
                                                          return_array=False)


class SurvivalTreeContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        # y = Surv.from_dataframe(experiment.)
        # CoxPHRegression = ModelWrapperCoxPH()

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "max_features": [1.0, "sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 7, 9, 10],
            "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        tune_distributions = {
            "max_depth": IntUniformDistribution(1, 16),
            "max_features": UniformDistribution(0.4, 1),
            "min_samples_leaf": IntUniformDistribution(2, 6),
            "min_samples_split": IntUniformDistribution(2, 10),
        }


        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="survival_tree",
            name="Survival Tree",
            class_def=SurvivalTreeWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class RandomSurvivalForestWrapper(RandomSurvivalForest):
    """A random survival forest.
    A random survival forest is a meta estimator that fits a number of
    survival trees on various sub-samples of the dataset and uses
    averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original input sample
    size but the samples are drawn with replacement if
    `bootstrap=True` (default).
    In each survival tree, the quality of a split is measured by the
    log-rank splitting rule.
    See the :ref:`User Guide </user_guide/random-survival-forest.ipynb>`,
    [1]_ and [2]_ for further description.
    Parameters
    ----------
    n_estimators : integer, optional, default: 100
        The number of trees in the forest.
    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    bootstrap : boolean, optional, default: True
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.
    oob_score : bool, default: False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional, default: 0
        Controls the verbosity when fitting and predicting.
    warm_start : bool, optional, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    max_samples : int or float, optional, default: None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
        `max_samples` should be in the interval `(0.0, 1.0]`.
    Attributes
    ----------
    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators.
    event_times_ : array of shape = (n_event_times,)
        Unique time points where events occurred.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    oob_score_ : float
        Concordance index of the training dataset obtained
        using an out-of-bag estimate.
    See also
    --------
    sksurv.tree.SurvivalTree
        A single survival tree.
    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    Compared to scikit-learn's random forest models, :class:`RandomSurvivalForest`
    currently does not support controlling the depth of a tree based on the log-rank
    test statistics or it's associated p-value, i.e., the parameters
    `min_impurity_decrease` or `min_impurity_split` are absent.
    In addition, the `feature_importances_` attribute is not available.
    It is recommended to estimate feature importances via
    `permutation-based methods <https://eli5.readthedocs.io>`_.
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behavior during
    fitting, ``random_state`` has to be fixed.
    References
    ----------
    .. [1] Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008).
           Random survival forests. The Annals of Applied Statistics, 2(3), 841–860.
    .. [2] Ishwaran, H., Kogalur, U. B. (2007). Random survival forests for R.
           R News, 7(2), 25–31. https://cran.r-project.org/doc/Rnews/Rnews_2007-2.pdf.
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None,
                 **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        super().__init__(n_estimators=n_estimators,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start,
                         max_samples=max_samples,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes)


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        # print(X.columns)
        # print(y.columns)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv, sample_weight=sample_weight)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        # print(X.columns)
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)

    def predict_survival_function(self, X):
        return super().predict_survival_function(X.loc[:, X.columns != 'time'].to_numpy(copy=True), return_array=False)

    def _predict_cumulative_hazard_function(self, X):
        return super().predict_cumulative_hazard_function(X.loc[:, X.columns != 'time'].to_numpy(copy=True),
                                                          return_array=False)


class RandomSurvivalForestContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        from sksurv.linear_model import CoxPHSurvivalAnalysis

        # y = Surv.from_dataframe(experiment.)
        # CoxPHRegression = ModelWrapperCoxPH()

        args = {"random_state": experiment.seed,
                "n_jobs": experiment.n_jobs_param,
                }
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 30, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 7, 9, 10],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "max_features": UniformDistribution(0.4, 1),
        }
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="rsf",
            name="RandomSurvivalForest",
            class_def=RandomSurvivalForestWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class ExtraSurvivalTreesWrapper(ExtraSurvivalTrees):
    """An extremely random survival forest.
    This class implements a meta estimator that fits a number of randomized
    survival trees (a.k.a. extra-trees) on various sub-samples of the dataset
    and uses averaging to improve the predictive accuracy and control
    over-fitting. The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).
    In each randomized survival tree, the quality of a split is measured by
    the log-rank splitting rule.
    Compared to :class:`RandomSurvivalForest`, randomness goes one step
    further in the way splits are computed. As in
    :class:`RandomSurvivalForest`, a random subset of candidate features is
    used, but instead of looking for the most discriminative thresholds,
    thresholds are drawn at random for each candidate feature and the best of
    these randomly-generated thresholds is picked as the splitting rule.
    Parameters
    ----------
    n_estimators : integer, optional, default: 100
        The number of trees in the forest.
    max_depth : int or None, optional, default: None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional, default: 6
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional, default: 3
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional, default: 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, string or None, optional, default: None
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int or None, optional, default: None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    bootstrap : boolean, optional, default: True
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.
    oob_score : bool, default: False
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : int, optional, default: 0
        Controls the verbosity when fitting and predicting.
    warm_start : bool, optional, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    max_samples : int or float, optional, default: None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
        `max_samples` should be in the interval `(0.0, 1.0]`.
    Attributes
    ----------
    estimators_ : list of SurvivalTree instances
        The collection of fitted sub-estimators.
    event_times_ : array of shape = (n_event_times,)
        Unique time points where events occurred.
    n_features_in_ : int
        The number of features when ``fit`` is performed.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    oob_score_ : float
        Concordance index of the training dataset obtained
        using an out-of-bag estimate.
    See also
    --------
    sksurv.tree.SurvivalTree
        A single survival tree.
    """
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=6,
                 min_samples_leaf=3,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_samples=None,
                 **kwargs):

        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

        # super(ExtraSurvivalTrees, self).__init__(base_estimator=SurvivalTree(splitter='random'))
        super().__init__(n_estimators=n_estimators,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start,
                         max_samples=max_samples,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv, sample_weight=sample_weight)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)

    def predict_survival_function(self, X):
        return super().predict_survival_function(X.loc[:, X.columns != 'time'].to_numpy(copy=True), return_array=False)

    def _predict_cumulative_hazard_function(self, X):
        return super().predict_cumulative_hazard_function(X.loc[:, X.columns != 'time'].to_numpy(copy=True),
                                                          return_array=False)


class ExtraSurvivalTreesContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        args = {
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 30, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [2, 3, 4, 5, 6],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "max_features": UniformDistribution(0.4, 1),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="et_surv",
            name="ExtraSurvivalTrees",
            class_def=ExtraSurvivalTreesWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class GradientBoostingSurvivalWrapper(GradientBoostingSurvivalAnalysis):
    """Gradient-boosted Cox proportional hazard loss with
        regression trees as base learner.
        In each stage, a regression tree is fit on the negative gradient
        of the loss function.
        For more details on gradient boosting see [1]_ and [2]_. If `loss='coxph'`,
        the partial likelihood of the proportional hazards model is optimized as
        described in [3]_. If `loss='ipcwls'`, the accelerated failture time model with
        inverse-probability of censoring weighted least squares error is optimized as
        described in [4]_. When using a non-zero `dropout_rate`, regularization is
        applied during training following [5]_.
        See the :ref:`User Guide </user_guide/boosting.ipynb>` for examples.
        Parameters
        ----------
        loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
            loss function to be optimized. 'coxph' refers to partial likelihood loss
            of Cox's proportional hazards model. The loss 'squared' minimizes a
            squared regression loss that ignores predictions beyond the time of censoring,
            and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.
        learning_rate : float, optional, default: 0.1
            learning rate shrinks the contribution of each tree by `learning_rate`.
            There is a trade-off between learning_rate and n_estimators.
        n_estimators : int, default: 100
            The number of regression trees to create. Gradient boosting
            is fairly robust to over-fitting so a large number usually
            results in better performance.
        criterion : string, optional, default: 'friedman_mse'
            The function to measure the quality of a split. Supported criteria
            are "friedman_mse" for the mean squared error with improvement
            score by Friedman, "mse" for mean squared error, and "mae" for
            the mean absolute error. The default value of "friedman_mse" is
            generally the best as it can provide a better approximation in
            some cases.
        min_samples_split : integer, optional, default: 2
            The minimum number of samples required to split an internal node.
        min_samples_leaf : integer, optional, default: 1
            The minimum number of samples required to be at a leaf node.
        min_weight_fraction_leaf : float, optional, default: 0.
            The minimum weighted fraction of the input samples required to be at a
            leaf node.
        max_depth : integer, optional, default: 3
            maximum depth of the individual regression estimators. The maximum
            depth limits the number of nodes in the tree. Tune this parameter
            for best performance; the best value depends on the interaction
            of the input variables.
            Ignored if ``max_leaf_nodes`` is not None.
        min_impurity_decrease : float, optional, default: 0.
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            The weighted impurity decrease equation is the following::
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)
            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.
            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.
        random_state : int seed, RandomState instance, or None, default: None
            The seed of the pseudo random number generator to use when
            shuffling the data.
        max_features : int, float, string or None, optional, default: None
            The number of features to consider when looking for the best split:
              - If int, then consider `max_features` features at each split.
              - If float, then `max_features` is a percentage and
                `int(max_features * n_features)` features are considered at each
                split.
              - If "auto", then `max_features=n_features`.
              - If "sqrt", then `max_features=sqrt(n_features)`.
              - If "log2", then `max_features=log2(n_features)`.
              - If None, then `max_features=n_features`.
            Choosing `max_features < n_features` leads to a reduction of variance
            and an increase in bias.
            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.
        max_leaf_nodes : int or None, optional, default: None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
        subsample : float, optional, default: 1.0
            The fraction of samples to be used for fitting the individual regression
            trees. If smaller than 1.0, this results in Stochastic Gradient
            Boosting. `subsample` interacts with the parameter `n_estimators`.
            Choosing `subsample < 1.0` leads to a reduction of variance
            and an increase in bias.
        dropout_rate : float, optional, default: 0.0
            If larger than zero, the residuals at each iteration are only computed
            from a random subset of base learners. The value corresponds to the
            percentage of base learners that are dropped. In each iteration,
            at least one base learner is dropped. This is an alternative regularization
            to shrinkage, i.e., setting `learning_rate < 1.0`.
        verbose : int, default: 0
            Enable verbose output. If 1 then it prints progress and performance
            once in a while (the more trees the lower the frequency). If greater
            than 1 then it prints progress and performance for every tree.
        ccp_alpha : non-negative float, optional, default: 0.0.
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        Attributes
        ----------
        n_estimators_ : int
            The number of estimators as selected by early stopping (if
            ``n_iter_no_change`` is specified). Otherwise it is set to
            ``n_estimators``.
        feature_importances_ : ndarray, shape = (n_features,)
            The feature importances (the higher, the more important the feature).
        estimators_ : ndarray of DecisionTreeRegressor, shape = (n_estimators, 1)
            The collection of fitted sub-estimators.
        train_score_ : ndarray, shape = (n_estimators,)
            The i-th score ``train_score_[i]`` is the deviance (= loss) of the
            model at iteration ``i`` on the in-bag sample.
            If ``subsample == 1`` this is the deviance on the training data.
        oob_improvement_ : ndarray, shape = (n_estimators,)
            The improvement in loss (= deviance) on the out-of-bag samples
            relative to the previous iteration.
            ``oob_improvement_[0]`` is the improvement in
            loss of the first stage over the ``init`` estimator.
        n_features_in_ : int
            Number of features seen during ``fit``.
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during ``fit``. Defined only when `X`
            has feature names that are all strings.
        References
        ----------
        .. [1] J. H. Friedman, "Greedy function approximation: A gradient boosting machine,"
               The Annals of Statistics, 29(5), 1189–1232, 2001.
        .. [2] J. H. Friedman, "Stochastic gradient boosting,"
               Computational Statistics & Data Analysis, 38(4), 367–378, 2002.
        .. [3] G. Ridgeway, "The state of boosting,"
               Computing Science and Statistics, 172–181, 1999.
        .. [4] Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., van der Laan, M. J.,
               "Survival ensembles", Biostatistics, 7(3), 355-73, 2006.
        .. [5] K. V. Rashmi and R. Gilad-Bachrach,
               "DART: Dropouts meet multiple additive regression trees,"
               in 18th International Conference on Artificial Intelligence and Statistics,
               2015, 489–497.
        """

    def __init__(self, loss="coxph", learning_rate=0.1, n_estimators=100,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3,
                 min_impurity_decrease=0., random_state=None,
                 max_features=None, max_leaf_nodes=None,
                 subsample=1.0, dropout_rate=0.0,
                 verbose=0,
                 ccp_alpha=0.0,
                 **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.ccp_alpha = ccp_alpha
        super().__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        # print(X.columns)
        # print(y.columns)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv, sample_weight=sample_weight)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        # print(X.columns)
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class GradientBoostingSurvivalContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 100, 10, inclusive=True),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                # 0.3,
                # 0.4,
                # 0.5,
            ],
            "subsample": np_list_arange(0.2, 1, 0.1, inclusive=True),
            "min_samples_split": [2, 4, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                # 0.3,
                # 0.4,
                # 0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "subsample": UniformDistribution(0.2, 1),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "max_depth": IntUniformDistribution(1, 11),
            "max_features": UniformDistribution(0.4, 1),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id='gb_surv',
            name="GradientBoostingSurvival",
            class_def=GradientBoostingSurvivalWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class ComponentWiseGradientBoostingSurvivalWrapper(ComponentwiseGradientBoostingSurvivalAnalysis):
    r"""Gradient boosting with component-wise least squares as base learner.
        See the :ref:`User Guide </user_guide/boosting.ipynb>` and [1]_ for further description.
        Parameters
        ----------
        loss : {'coxph', 'squared', 'ipcwls'}, optional, default: 'coxph'
            loss function to be optimized. 'coxph' refers to partial likelihood loss
            of Cox's proportional hazards model. The loss 'squared' minimizes a
            squared regression loss that ignores predictions beyond the time of censoring,
            and 'ipcwls' refers to inverse-probability of censoring weighted least squares error.
        learning_rate : float, optional, default: 0.1
            learning rate shrinks the contribution of each base learner by `learning_rate`.
            There is a trade-off between `learning_rate` and `n_estimators`.
        n_estimators : int, default: 100
            The number of boosting stages to perform. Gradient boosting
            is fairly robust to over-fitting so a large number usually
            results in better performance.
        subsample : float, optional, default: 1.0
            The fraction of samples to be used for fitting the individual base
            learners. If smaller than 1.0 this results in Stochastic Gradient
            Boosting. `subsample` interacts with the parameter `n_estimators`.
            Choosing `subsample < 1.0` leads to a reduction of variance
            and an increase in bias.
        dropout_rate : float, optional, default: 0.0
            If larger than zero, the residuals at each iteration are only computed
            from a random subset of base learners. The value corresponds to the
            percentage of base learners that are dropped. In each iteration,
            at least one base learner is dropped. This is an alternative regularization
            to shrinkage, i.e., setting `learning_rate < 1.0`.
        random_state : int seed, RandomState instance, or None, default: None
            The seed of the pseudo random number generator to use when
            shuffling the data.
        verbose : int, default: 0
            Enable verbose output. If 1 then it prints progress and performance
            once in a while (the more trees the lower the frequency). If greater
            than 1 then it prints progress and performance for every tree.
        Attributes
        ----------
        coef_ : array, shape = (n_features + 1,)
            The aggregated coefficients. The first element `coef\_[0]` corresponds
            to the intercept. If loss is `coxph`, the intercept will always be zero.
        loss_ : LossFunction
            The concrete ``LossFunction`` object.
        estimators_ : list of base learners
            The collection of fitted sub-estimators.
        train_score_ : array, shape = (n_estimators,)
            The i-th score ``train_score_[i]`` is the deviance (= loss) of the
            model at iteration ``i`` on the in-bag sample.
            If ``subsample == 1`` this is the deviance on the training data.
        oob_improvement_ : array, shape = (n_estimators,)
            The improvement in loss (= deviance) on the out-of-bag samples
            relative to the previous iteration.
            ``oob_improvement_[0]`` is the improvement in
            loss of the first stage over the ``init`` estimator.
        n_features_in_ : int
            Number of features seen during ``fit``.
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during ``fit``. Defined only when `X`
            has feature names that are all strings.
        References
        ----------
        .. [1] Hothorn, T., Bühlmann, P., Dudoit, S., Molinaro, A., van der Laan, M. J.,
               "Survival ensembles", Biostatistics, 7(3), 355-73, 2006
        """

    def __init__(self, loss="coxph", learning_rate=0.1, n_estimators=100, subsample=1.0,
                 dropout_rate=0, random_state=None, verbose=0, **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.verbose = verbose
        # self.survival_train = None
        super().__init__(loss=loss, n_estimators=n_estimators, learning_rate=learning_rate,
                         subsample=subsample, dropout_rate=dropout_rate, random_state=random_state,
                         verbose=verbose)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv, sample_weight=sample_weight)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class ComponentWiseGradientBoostingSurvivalContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        self.survival_train = None


        args = {
            "random_state": experiment.seed,
        }
        tune_args = {}
        tune_grid = {
            "loss" : ["coxph", "squared", "ipcwls"],
            "learning_rate": [0.1, 0.05, 0.01, 0.005, 0.001],
            "n_estimators": [100, 200, 500, 1000, 2000],
            "subsample": [0.5, 0.75, 1.0],
            "dropout_rate": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id='cwgb_surv',
            name="ComponentWiseGradientBoostingSurvival",
            class_def=ComponentWiseGradientBoostingSurvivalWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class HingeLossSVMSurvivalContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        self.survival_train = None


        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": [1.0],
            "solver": ["ecos", "osqp"],
            "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed", "cosine"],
            "degree": [1, 2, 3, 4, 5],
            "pairs": ["all", "nearest", "next"],
        }
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id='hloss_svm',
            name="HingeLossSVMSurvival",
            class_def=HingeLossSVMSurvivalWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class HingeLossSVMSurvivalWrapper(HingeLossSurvivalSVM):
    """Naive implementation of kernel survival support vector machine.
    A new set of samples is created by building the difference between any two feature
    vectors in the original data, thus this version requires :math:`O(\\text{n_samples}^4)` space and
    :math:`O(\\text{n_samples}^6 \\cdot \\text{n_features})`.
    See :class:`sksurv.svm.NaiveSurvivalSVM` for the linear naive survival SVM based on liblinear.
    .. math::
          \\min_{\\mathbf{w}}\\quad
          \\frac{1}{2} \\lVert \\mathbf{w} \\rVert_2^2
          + \\gamma \\sum_{i = 1}^n \\xi_i \\\\
          \\text{subject to}\\quad
          \\mathbf{w}^\\top \\phi(\\mathbf{x})_i - \\mathbf{w}^\\top \\phi(\\mathbf{x})_j \\geq 1 - \\xi_{ij},\\quad
          \\forall (i, j) \\in \\mathcal{P}, \\\\
          \\xi_i \\geq 0,\\quad \\forall (i, j) \\in \\mathcal{P}.
          \\mathcal{P} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1 \\}_{i,j=1,\\dots,n}.
    See [1]_, [2]_, [3]_ for further description.
    Parameters
    ----------
    solver : "ecos" | "osqp", optional, default: ecos
        Which quadratic program solver to use.
    alpha : float, positive, default: 1
        Weight of penalizing the hinge loss in the objective function.
    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel.
        Default: "linear"
    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: ``1/n_features``.
        Ignored by other kernels.
    degree : int, default: 3
        Degree for poly kernels. Ignored by other kernels.
    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as call
    pairs : "all" | "nearest" | "next", optional, default: "all"
        Which constraints to use in the optimization problem.
        - all: Use all comparable pairs. Scales quadratic in number of samples.
        - nearest: Only considers comparable pairs :math:`(i, j)` where :math:`j` is the
          uncensored sample with highest survival time smaller than :math:`y_i`.
          Scales linear in number of samples (cf. :class:`sksurv.svm.MinlipSurvivalSVM`).
        - next: Only compare against direct nearest neighbor according to observed time,
          disregarding its censoring status. Scales linear in number of samples.
    verbose : bool, default: False
        Enable verbose output of solver.
    timeit : False or int
        If non-zero value is provided the time it takes for optimization is measured.
        The given number of repetitions are performed. Results can be accessed from the
        ``timings_`` attribute.
    max_iter : int, optional
        Maximum number of iterations to perform. By default
        use solver's default value.
    Attributes
    ----------
    X_fit_ : ndarray
        Training data.
    coef_ : ndarray, shape = (n_samples,)
        Coefficients of the features in the decision function.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    References
    ----------
    .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A., & Van Huffel, S.
           Support Vector Machines for Survival Analysis. In Proc. of the 3rd Int. Conf.
           on Computational Intelligence in Medicine and Healthcare (CIMED). 1-8. 2007
    .. [2] Evers, L., Messow, C.M.,
           "Sparse kernel methods for high-dimensional survival data",
           Bioinformatics 24(14), 1632-8, 2008.
    .. [3] Van Belle, V., Pelckmans, K., Suykens, J.A., Van Huffel, S.,
           "Survival SVM: a practical scalable algorithm",
           In: Proc. of 16th European Symposium on Artificial Neural Networks,
           89-94, 2008.
    """

    def __init__(self, solver="ecos",
                 alpha=1.0, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None,
                 pairs="all", verbose=False, timeit=None, max_iter=None, **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        self.solver = solver
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.pairs = pairs
        self.verbose = verbose
        self.timeit = timeit
        self.max_iter = max_iter
        super().__init__(solver=solver, alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,
                         kernel_params=kernel_params, pairs=pairs, verbose=verbose, timeit=timeit, max_iter=max_iter)


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class FastKernelSVMSurvivalContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        self.survival_train = None


        args = {
            "random_state": experiment.seed,
        }
        tune_args = {}
        tune_grid = {
            "alpha": [1.0],
            "rank_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "fit_intercept": [True, False],
            "kernel": ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
            # "gamma": [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "degree": [1, 2, 3, 4, 5],
            # "coef0": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "max_iter": [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            # "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "optimizer": ["avltree", "rbtree"],
        }
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id='fastk_svm',
            name="FastKernelSVMSurvival",
            class_def=FastKernelSVMSurvivalWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class FastKernelSVMSurvivalWrapper(FastKernelSurvivalSVM):
    """Efficient Training of kernel Survival Support Vector Machine.
        See the :ref:`User Guide </user_guide/survival-svm.ipynb>` and [1]_ for further description.
        Parameters
        ----------
        alpha : float, positive, default: 1
            Weight of penalizing the squared hinge loss in the objective function
        rank_ratio : float, optional, default: 1.0
            Mixing parameter between regression and ranking objective with ``0 <= rank_ratio <= 1``.
            If ``rank_ratio = 1``, only ranking is performed, if ``rank_ratio = 0``, only regression
            is performed. A non-zero value is only allowed if optimizer is one of 'avltree', 'PRSVM',
            or 'rbtree'.
        fit_intercept : boolean, optional, default: False
            Whether to calculate an intercept for the regression model. If set to ``False``, no intercept
            will be calculated. Has no effect if ``rank_ratio = 1``, i.e., only ranking is performed.
        kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
            Kernel.
            Default: "linear"
        degree : int, default: 3
            Degree for poly kernels. Ignored by other kernels.
        gamma : float, optional
            Kernel coefficient for rbf and poly kernels. Default: ``1/n_features``.
            Ignored by other kernels.
        coef0 : float, optional
            Independent term in poly and sigmoid kernels.
            Ignored by other kernels.
        kernel_params : mapping of string to any, optional
            Parameters (keyword arguments) and values for kernel passed as call
        max_iter : int, optional, default: 20
            Maximum number of iterations to perform in Newton optimization
        verbose : bool, optional, default: False
            Whether to print messages during optimization
        tol : float, optional
            Tolerance for termination. For detailed control, use solver-specific
            options.
        optimizer : "avltree" | "rbtree", optional, default: "rbtree"
            Which optimizer to use.
        random_state : int or :class:`numpy.random.RandomState` instance, optional
            Random number generator (used to resolve ties in survival times).
        timeit : False or int
            If non-zero value is provided the time it takes for optimization is measured.
            The given number of repetitions are performed. Results can be accessed from the
            ``optimizer_result_`` attribute.
        Attributes
        ----------
        coef_ : ndarray, shape = (n_samples,)
            Weights assigned to the samples in training data to represent
            the decision function in kernel space.
        fit_X_ : ndarray
            Training data.
        optimizer_result_ : :class:`scipy.optimize.optimize.OptimizeResult`
            Stats returned by the optimizer. See :class:`scipy.optimize.optimize.OptimizeResult`.
        n_features_in_ : int
            Number of features seen during ``fit``.
        feature_names_in_ : ndarray of shape (`n_features_in_`,)
            Names of features seen during ``fit``. Defined only when `X`
            has feature names that are all strings.
        See also
        --------
        FastSurvivalSVM
            Fast implementation for linear kernel.
        References
        ----------
        .. [1] Pölsterl, S., Navab, N., and Katouzian, A.,
               *An Efficient Training Algorithm for Kernel Survival Support Vector Machines*
               4th Workshop on Machine Learning in Life Sciences,
               23 September 2016, Riva del Garda, Italy. arXiv:1611.07054
        """

    def __init__(self, alpha=1, rank_ratio=1.0, fit_intercept=False, kernel="rbf",
                 gamma=None, degree=3, coef0=1, kernel_params=None, max_iter=20, verbose=False, tol=None,
                 optimizer=None, random_state=None, timeit=False, **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        super().__init__(alpha=alpha, rank_ratio=rank_ratio, fit_intercept=fit_intercept,
                         max_iter=max_iter, verbose=verbose, tol=tol,
                         optimizer=optimizer, random_state=random_state,
                         timeit=timeit)
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.fit_intercept = fit_intercept
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.optimizer = optimizer
        self.random_state = random_state
        self.timeit = timeit

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class FastSVMSurvivalWrapper(FastSurvivalSVM):
    """Efficient Training of linear Survival Support Vector Machine
    Training data consists of *n* triplets :math:`(\\mathbf{x}_i, y_i, \\delta_i)`,
    where :math:`\\mathbf{x}_i` is a *d*-dimensional feature vector, :math:`y_i > 0`
    the survival time or time of censoring, and :math:`\\delta_i \\in \\{0,1\\}`
    the binary event indicator. Using the training data, the objective is to
    minimize the following function:
    .. math::
         \\arg \\min_{\\mathbf{w}, b} \\frac{1}{2} \\mathbf{w}^\\top \\mathbf{w}
         + \\frac{\\alpha}{2} \\left[ r \\sum_{i,j \\in \\mathcal{P}}
         \\max(0, 1 - (\\mathbf{w}^\\top \\mathbf{x}_i - \\mathbf{w}^\\top \\mathbf{x}_j))^2
         + (1 - r) \\sum_{i=0}^n \\left( \\zeta_{\\mathbf{w}, b} (y_i, x_i, \\delta_i)
         \\right)^2 \\right]
        \\zeta_{\\mathbf{w},b} (y_i, \\mathbf{x}_i, \\delta_i) =
        \\begin{cases}
        \\max(0, y_i - \\mathbf{w}^\\top \\mathbf{x}_i - b) \\quad \\text{if $\\delta_i = 0$,} \\\\
        y_i - \\mathbf{w}^\\top \\mathbf{x}_i - b \\quad \\text{if $\\delta_i = 1$,} \\\\
        \\end{cases}
        \\mathcal{P} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1 \\}_{i,j=1,\\dots,n}
    The hyper-parameter :math:`\\alpha > 0` determines the amount of regularization
    to apply: a smaller value increases the amount of regularization and a
    higher value reduces the amount of regularization. The hyper-parameter
    :math:`r \\in [0; 1]` determines the trade-off between the ranking objective
    and the regresson objective. If :math:`r = 1` it reduces to the ranking
    objective, and if :math:`r = 0` to the regression objective. If the regression
    objective is used, survival/censoring times are log-transform and thus cannot be
    zero or negative.
    See the :ref:`User Guide </user_guide/survival-svm.ipynb>` and [1]_ for further description.
    Parameters
    ----------
    alpha : float, positive, default: 1
        Weight of penalizing the squared hinge loss in the objective function
    rank_ratio : float, optional, default: 1.0
        Mixing parameter between regression and ranking objective with ``0 <= rank_ratio <= 1``.
        If ``rank_ratio = 1``, only ranking is performed, if ``rank_ratio = 0``, only regression
        is performed. A non-zero value is only allowed if optimizer is one of 'avltree', 'rbtree',
        or 'direct-count'.
    fit_intercept : boolean, optional, default: False
        Whether to calculate an intercept for the regression model. If set to ``False``, no intercept
        will be calculated. Has no effect if ``rank_ratio = 1``, i.e., only ranking is performed.
    max_iter : int, optional, default: 20
        Maximum number of iterations to perform in Newton optimization
    verbose : bool, optional, default: False
        Whether to print messages during optimization
    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
    optimizer : "avltree" | "direct-count" | "PRSVM" | "rbtree" | "simple", optional, default: avltree
        Which optimizer to use.
    random_state : int or :class:`numpy.random.RandomState` instance, optional
        Random number generator (used to resolve ties in survival times).
    timeit : False or int
        If non-zero value is provided the time it takes for optimization is measured.
        The given number of repetitions are performed. Results can be accessed from the
        ``optimizer_result_`` attribute.
    Attributes
    ----------
    coef_ : ndarray, shape = (n_features,)
        Coefficients of the features in the decision function.
    optimizer_result_ : :class:`scipy.optimize.optimize.OptimizeResult`
        Stats returned by the optimizer. See :class:`scipy.optimize.optimize.OptimizeResult`.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    See also
    --------
    FastKernelSurvivalSVM
        Fast implementation for arbitrary kernel functions.
    References
    ----------
    .. [1] Pölsterl, S., Navab, N., and Katouzian, A.,
           "Fast Training of Support Vector Machines for Survival Analysis",
           Machine Learning and Knowledge Discovery in Databases: European Conference,
           ECML PKDD 2015, Porto, Portugal,
           Lecture Notes in Computer Science, vol. 9285, pp. 243-259 (2015)
    """
    def __init__(self, alpha=1, rank_ratio=1.0, fit_intercept=False,
                 max_iter=20, verbose=False, tol=None,
                 optimizer=None, random_state=None, timeit=False, **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        super().__init__(alpha=alpha, rank_ratio=rank_ratio, fit_intercept=fit_intercept,
                         max_iter=max_iter, verbose=verbose, tol=tol,
                         optimizer=optimizer, random_state=random_state,
                         timeit=timeit)
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.tol = tol
        self.optimizer = optimizer
        self.random_state = random_state
        self.timeit = timeit

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class FastSVMSurvivalContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        self.survival_train = None


        args = {
            "random_state": experiment.seed,
        }
        tune_args = {}
        tune_grid = {
            "alpha": [1.0],
            # "rank_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "fit_intercept": [True, False],
            "max_iter": [None, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            # "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "optimizer": ["avltree", "direct-count", "PRSVM", "rbtree", "simple"]
        }
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id='fast_svm',
            name="FastSVMSurvival",
            class_def=FastSVMSurvivalWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class MinlipSVMSurvivalWrapper(MinlipSurvivalAnalysis):
    """Survival model related to survival SVM, using a minimal Lipschitz smoothness strategy
    instead of a maximal margin strategy.
    .. math::
          \\min_{\\mathbf{w}}\\quad
          \\frac{1}{2} \\lVert \\mathbf{w} \\rVert_2^2
          + \\gamma \\sum_{i = 1}^n \\xi_i \\\\
          \\text{subject to}\\quad
          \\mathbf{w}^\\top \\mathbf{x}_i - \\mathbf{w}^\\top \\mathbf{x}_j \\geq y_i - y_j - \\xi_i,\\quad
          \\forall (i, j) \\in \\mathcal{P}_\\text{1-NN}, \\\\
          \\xi_i \\geq 0,\\quad \\forall i = 1,\\dots,n.
          \\mathcal{P}_\\text{1-NN} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1
          \\land \\nexists k : y_i > y_k > y_j \\land \\delta_k = 1 \\}_{i,j=1}^n.
    See [1]_ for further description.
    Parameters
    ----------
    solver : "ecos" | "osqp", optional, default: ecos
        Which quadratic program solver to use.
    alpha : float, positive, default: 1
        Weight of penalizing the hinge loss in the objective function.
    kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel.
        Default: "linear"
    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: ``1/n_features``.
        Ignored by other kernels.
    degree : int, default: 3
        Degree for poly kernels. Ignored by other kernels.
    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.
    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as call
    pairs : "all" | "nearest" | "next", optional, default: "nearest"
        Which constraints to use in the optimization problem.
        - all: Use all comparable pairs. Scales quadratic in number of samples
          (cf. :class:`sksurv.svm.HingeLossSurvivalSVM`).
        - nearest: Only considers comparable pairs :math:`(i, j)` where :math:`j` is the
          uncensored sample with highest survival time smaller than :math:`y_i`.
          Scales linear in number of samples.
        - next: Only compare against direct nearest neighbor according to observed time,
          disregarding its censoring status. Scales linear in number of samples.
    verbose : bool, default: False
        Enable verbose output of solver
    timeit : False or int
        If non-zero value is provided the time it takes for optimization is measured.
        The given number of repetitions are performed. Results can be accessed from the
        ``timings_`` attribute.
    max_iter : int, optional
        Maximum number of iterations to perform. By default
        use solver's default value.
    Attributes
    ----------
    X_fit_ : ndarray
        Training data.
    coef_ : ndarray, shape = (n_samples,)
        Coefficients of the features in the decision function.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when `X`
        has feature names that are all strings.
    References
    ----------
    .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A. K., and Van Huffel, S.
           Learning transformation models for ranking and survival analysis.
           The Journal of Machine Learning Research, 12, 819-862. 2011
    """

    def __init__(self, solver="ecos",
                 alpha=1.0, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None,
                 pairs="nearest", verbose=False, timeit=None, max_iter=None, **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        self.solver = solver
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.pairs = pairs
        self.verbose = verbose
        self.timeit = timeit
        self.max_iter = max_iter
        super().__init__(solver=solver,
                         alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0,
                         kernel_params=kernel_params, pairs=pairs, verbose=verbose, timeit=timeit, max_iter=max_iter)


    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class MinlipSVMSurvivalContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        self.survival_train = None

        args = {}
        tune_args = {}
        tune_grid = {
            "alpha": [1.0],
            "solver": ["ecos", "osqp"],
            "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed", "cosine"],
            "degree": [1, 2, 3, 4, 5],
            "pairs": ["all", "nearest", "next"],
        }
        tune_distributions = {}
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id='minlip_svm',
            name="MinlipSVMSurvival",
            class_def=MinlipSVMSurvivalWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


class NaiveSVMSurvivalWrapper(NaiveSurvivalSVM):
    """Naive version of linear Survival Support Vector Machine.
       Uses regular linear support vector classifier (liblinear).
       A new set of samples is created by building the difference between any two feature
       vectors in the original data, thus this version requires `O(n_samples^2)` space.
       See :class:`sksurv.svm.HingeLossSurvivalSVM` for the kernel naive survival SVM.
       .. math::
             \\min_{\\mathbf{w}}\\quad
             \\frac{1}{2} \\lVert \\mathbf{w} \\rVert_2^2
             + \\gamma \\sum_{i = 1}^n \\xi_i \\\\
             \\text{subject to}\\quad
             \\mathbf{w}^\\top \\mathbf{x}_i - \\mathbf{w}^\\top \\mathbf{x}_j \\geq 1 - \\xi_{ij},\\quad
             \\forall (i, j) \\in \\mathcal{P}, \\\\
             \\xi_i \\geq 0,\\quad \\forall (i, j) \\in \\mathcal{P}.
             \\mathcal{P} = \\{ (i, j) \\mid y_i > y_j \\land \\delta_j = 1 \\}_{i,j=1,\\dots,n}.
       See [1]_, [2]_ for further description.
       Parameters
       ----------
       alpha : float, positive, default: 1.0
           Weight of penalizing the squared hinge loss in the objective function.
       loss : string, 'hinge' or 'squared_hinge', default: 'squared_hinge'
           Specifies the loss function. 'hinge' is the standard SVM loss
           (used e.g. by the SVC class) while 'squared_hinge' is the
           square of the hinge loss.
       penalty : 'l1' | 'l2', default: 'l2'
           Specifies the norm used in the penalization. The 'l2'
           penalty is the standard used in SVC. The 'l1' leads to `coef_`
           vectors that are sparse.
       dual : bool, default: True
           Select the algorithm to either solve the dual or primal
           optimization problem. Prefer dual=False when n_samples > n_features.
       tol : float, optional, default: 1e-4
           Tolerance for stopping criteria.
       verbose : int, default: 0
           Enable verbose output. Note that this setting takes advantage of a
           per-process runtime setting in liblinear that, if enabled, may not work
           properly in a multithreaded context.
       random_state : int seed, RandomState instance, or None, default: None
           The seed of the pseudo random number generator to use when
           shuffling the data.
       max_iter : int, default: 1000
           The maximum number of iterations to be run.
       See also
       --------
       sksurv.svm.FastSurvivalSVM
           Alternative implementation with reduced time complexity for training.
       References
       ----------
       .. [1] Van Belle, V., Pelckmans, K., Suykens, J. A., & Van Huffel, S.
              Support Vector Machines for Survival Analysis. In Proc. of the 3rd Int. Conf.
              on Computational Intelligence in Medicine and Healthcare (CIMED). 1-8. 2007
       .. [2] Evers, L., Messow, C.M.,
              "Sparse kernel methods for high-dimensional survival data",
              Bioinformatics 24(14), 1632-8, 2008.
       """

    def __init__(self, penalty='l2', loss='squared_hinge', dual=False, tol=1e-4,
                 alpha=1.0, verbose=0, random_state=None, max_iter=1000, **kwargs):
        # Filter out any extra kwargs that the parent class doesn't accept
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_score']}
        
        super().__init__(penalty=penalty,
                         loss=loss,
                         dual=dual,
                         tol=tol,
                         verbose=verbose,
                         random_state=random_state,
                         max_iter=max_iter)
        self.alpha = alpha
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().fit(X.loc[:, X.columns != 'time'].to_numpy(copy=True), y_surv)

    def predict(self, X: pd.DataFrame, check_input: Optional[bool] = True):
        return super().predict(X.loc[:, X.columns != 'time'].to_numpy(copy=True))

    def score(self, X, y):
        y_time = X["time"].values.ravel()
        y_surv = Surv.from_arrays(event=y, time=y_time)
        return super().score(X, y_surv)


class NaiveSVMSurvivalContainer(SurvivalAnalysisContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False
        self.survival_train = None

        args = {
            "random_state": experiment.seed,
        }
        tune_args = {}
        tune_grid = {
            "alpha": [1.0],
            "loss": ["squared_hinge", "hinge"],
            "penalty": ["l1", "l2"],
            "dual": [True, False],
            "tol": [1e-4],
        }
        tune_distributions = {}
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id='naive_svm',
            name="NaiveSVMSurvival",
            class_def=NaiveSVMSurvivalWrapper,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            is_gpu_enabled=gpu_imported,
            shap=False,
        )


def get_all_model_containers(
    experiment: Any, raise_errors: bool = True
) -> Dict[str, SurvivalAnalysisContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), experiment, SurvivalAnalysisContainer, raise_errors
    )
