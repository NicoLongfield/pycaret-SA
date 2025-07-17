# Module: containers.metrics.regression
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# The purpose of this module is to serve as a central repository of regression metrics. The `regression` module will
# call `get_all_metrics_containers()`, which will return instances of all classes in this module that have `RegressionMetricContainer`
# as a base (but not `RegressionMetricContainer` itself). In order to add a new model, you only need to create a new class that has
# `RegressionMetricContainer` as a base, set all of the required parameters in the `__init__` and then call `super().__init__`
# to complete the process. Refer to the existing classes for examples.

from typing import Any, Dict, Optional, Union

import numpy as np
import sksurv.metrics
from sklearn import metrics
from sklearn.metrics._regression import _check_reg_targets
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.validation import check_consistent_length

import pycaret.containers.base_container
import pycaret.internal.metrics
from pycaret.containers.metrics.base_metric import MetricContainer

from sksurv.functions import StepFunction
import sksurv.metrics as metrics_sksurv
class RegressionMetricContainer(MetricContainer):
    """
    Base regression metric container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full name.
    score_func : type
        The callable used for the score function, eg. sklearn.metrics.accuracy_score.
    scorer : str or callable, default = None
        The scorer passed to models. Can be a string representing a built-in sklearn scorer,
        a sklearn Scorer object, or None, in which case a Scorer object will be created from
        score_func and args.
    target : str, default = 'pred'
        The target of the score function. Only 'pred' is supported for regression.
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing score_func of class_def class.
    display_name : str, default = None
        Display name (shorter than name). Used in display dataframe header. If None or empty, will use name.
    greater_is_better: bool, default = True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    is_custom : bool, default = False
        Is the metric custom. Should be False for all metrics defined in PyCaret.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full name.
    score_func : type
        The callable used for the score function, eg. metrics.accuracy_score.
    scorer : str or callable
        The scorer passed to models. Can be a string representing a built-in sklearn scorer,
        a sklearn Scorer object, or None, in which case a Scorer object will be created from
        score_func and args.
    target : str
        The target of the score function.
        - 'pred' for the prediction table
    args : dict
        The arguments to always pass to constructor when initializing score_func of class_def class.
    display_name : str
        Display name (shorter than name). Used in display dataframe header.
    greater_is_better: bool
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    is_custom : bool
        Is the metric custom. Should be False for all metrics defined in PyCaret.

    """

    def __init__(
        self,
        id: str,
        name: str,
        score_func: type,
        scorer: Optional[Union[str, _BaseScorer]] = None,
        target: str = "pred",
        args: Dict[str, Any] = None,
        display_name: Optional[str] = None,
        greater_is_better: bool = True,
        is_custom: bool = False,
    ) -> None:

        allowed_targets = ["pred"]
        if not target in allowed_targets:
            raise ValueError(f"Target must be one of {', '.join(allowed_targets)}.")

        if not args:
            args = {}
        if not isinstance(args, dict):
            raise TypeError("args needs to be a dictionary.")

        scorer = (
            scorer
            if scorer
            else pycaret.internal.metrics.make_scorer_with_error_score(
                score_func,
                greater_is_better=greater_is_better,
                error_score=0.0,
                **args,
            )
        )

        super().__init__(
            id=id,
            name=name,
            score_func=score_func,
            scorer=scorer,
            args=args,
            display_name=display_name,
            greater_is_better=greater_is_better,
            is_custom=is_custom,
        )

        self.target = target

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
        d = {
            "ID": self.id,
            "Name": self.name,
            "Display Name": self.display_name,
            "Score Function": self.score_func,
            "Scorer": self.scorer,
            "Target": self.target,
            "Args": self.args,
            "Greater is Better": self.greater_is_better,
            "Custom": self.is_custom,
        }

        return d

from sklearn.metrics._scorer import _Scorer

#
# class SurvScorer(_Scorer):
#
#     def _score(self, method_caller, estimator, X, y):
#         """Score the given test data.
#         Parameters
#         ----------
#         estimator : object
#             Trained estimator to use for scoring. Must have a predict method.
#         X : array-like or sparse matrix
#             Test data that will be fed to the estimator.
#         y : array-like
#             Gold standard target values for X.
#         sample_weight : array-like, optional (default=None)
#             Sample weights.
#         Returns
#         -------
#         score : float
#             Score function applied to prediction of estimator on X.
#         """
#
#         y_pred = method_caller(estimator, "score", X, y)
#         # if sample_weight is not None:
#         #     sample_weight = np.array(sample_weight)
#         #     sample_weight = np.sqrt(sample_weight)
#         #     # y_pred = y_pred * sample_weight
#         #     y = y * sample_weight
#         return y_pred # estimator.score(X, y)

from sksurv.util import Surv
class SurvScorer(_Scorer):

    def _score(self, method_caller, estimator, X, y, sample_weight=None):
        """Score the given test data.
        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict method.
        X : array-like or sparse matrix
            Test data that will be fed to the estimator.
        y : array-like
            Gold standard target values for X.
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_time = X["time"].values.ravel()
        survival_test = Surv.from_arrays(event=y, time=y_time)
        estimate = method_caller(estimator, "predict", X)
        survival_train = self._kwargs["survival_train"]
        train_time = survival_train["time"]
        train_event = survival_train["event"]
        time_range = self._kwargs["time_range"]
        lower, upper = np.percentile(y_time, [20, 60])
        diff = (upper - lower)*0.25
        prob_time = np.linspace(lower+diff, upper-diff, len(y_time), endpoint=False)

        if self._score_func.__name__ == "concordance_index_censored":
            # scores = self._score_func(np.array(y).astype(bool), y_time, estimate, tied_tol=1e-8)
            # score = scores[0]
            score = estimator.score(X, np.array(y).astype(bool))

        elif self._score_func.__name__ == "concordance_index_ipcw":
            scores = self._score_func(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8)
            score = scores[0]

        else:
            if not hasattr(estimator, 'predict_survival_function'):
                score = 0
            else:
                if self._score_func.__name__ == "integrated_brier_score":

                    prob = np.row_stack([
                        fn(prob_time) for fn in method_caller(estimator, 'predict_survival_function', X)
                    ])


                    scores = self._score_func(survival_train, survival_test, prob, prob_time)
                    score = scores
                if self._score_func.__name__ == "cumulative_dynamic_auc":
                    if hasattr(estimator, '_predict_cumulative_hazard_function'):
                        prob = np.row_stack([
                            fn(prob_time) for fn in method_caller(estimator, '_predict_cumulative_hazard_function', X)
                        ])
                    else:
                        prob = np.row_stack([
                            fn(prob_time) for fn in method_caller(estimator, 'predict_cumulative_hazard_function', X)
                        ])
                    scores = self._score_func(survival_train, survival_test, prob, prob_time)
                    score = scores[1]

        return score


class SkSurvScoreFuncPatch:
    def __init__(self, score_func):
        self._score_func = score_func
        self.score_func_name = self._score_func.__name__
        if self._score_func.__name__ == "concordance_index_censored":
            self.is_cindex_censored = True
        else:
            self.is_cindex_censored = False
        if self._score_func.__name__ == "integrated_brier_score" or\
                self._score_func.__name__ == "cumulative_dynamic_auc":
            self.need_survival_function = True
        else:
            self.need_survival_function = False


    def __call__(self, y_test, y_train, X_train, X_test, target, **kwargs):
        y_time = X_test["time"].values.ravel()
        survival_train = Surv.from_arrays(event=np.array(y_train).astype(bool), time=X_train["time"].values.ravel())
        survival_test = Surv.from_arrays(event=np.array(y_test).astype(bool), time=y_time)

        if self.is_cindex_censored:
            y = np.array(y_test).astype(bool)
            scores = self._score_func(np.array(y).astype(bool), y_time, target)
            score = scores[0]
        else:
            if self.need_survival_function==False:
                scores = self._score_func(survival_train, survival_test, target, tau=None, tied_tol=1e-8)
                score = scores[0]
            else:
                # lower, upper = np.percentile(y_time, [10, 90])
                # prob_time = np.arange(lower, upper+1)
                lower, upper = np.percentile(y_time, [20, 60])
                diff = (upper - lower) * 0.25
                prob_time = np.linspace(lower + diff, upper - diff, len(y_time), endpoint=False)

                pred_surv = kwargs["pred_surv"] if "pred_surv" in kwargs else None
                pred_hazard = kwargs["pred_hazard"] if "pred_hazard" in kwargs else None
                if pred_surv is not None:
                    if self.score_func_name == "cumulative_dynamic_auc":
                        prob_hazard = np.row_stack([
                            fn(prob_time) for fn in pred_hazard
                        ])
                        scores = self._score_func(survival_train, survival_test, prob_hazard, prob_time)
                        score = scores[1]
                    else:
                        prob_surv = np.row_stack([
                            fn(prob_time) for fn in pred_surv
                        ])
                        scores = self._score_func(survival_train, survival_test, prob_surv, prob_time)
                        score = scores
                else:
                    score = 0
        return score



class CICensoredMetricContainer(RegressionMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        scorer_args_keys = ['survival_train', 'time_range']
        super().__init__(
            id="ci-cens",
            name="C-IC",
            score_func=SkSurvScoreFuncPatch(sksurv.metrics.concordance_index_censored),
            greater_is_better=True,
            scorer=SurvScorer(sksurv.metrics.concordance_index_censored, 1, {k: globals_dict[k] for k in scorer_args_keys}),
            # args={"pred_surv": Optional[StepFunction]},
        )


class CIIPCWMetricContainer(RegressionMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        scorer_args_keys = ['survival_train', 'time_range']
        super().__init__(
            id="ci-ipcw",
            name="C-II",
            score_func=SkSurvScoreFuncPatch(sksurv.metrics.concordance_index_ipcw),
            greater_is_better=True,
            scorer=SurvScorer(sksurv.metrics.concordance_index_ipcw, 1, {k: globals_dict[k] for k in scorer_args_keys}),
        )

class IBSMetricContainer(RegressionMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        scorer_args_keys = ['survival_train', 'time_range']
        super().__init__(
            id="ibs",
            name="IBS",
            score_func=SkSurvScoreFuncPatch(sksurv.metrics.integrated_brier_score),
            greater_is_better=True, # It's false in the original implementation
            scorer=SurvScorer(sksurv.metrics.integrated_brier_score, 1, {k: globals_dict[k] for k in scorer_args_keys}),
        )


class CumulativeDynamicAUCContainer(RegressionMetricContainer):
    def __init__(self, globals_dict: dict) -> None:
        scorer_args_keys = ['survival_train', 'time_range']
        super().__init__(
            id="CAUC",
            name="CAUC",
            score_func=SkSurvScoreFuncPatch(sksurv.metrics.cumulative_dynamic_auc),
            greater_is_better=True,
            scorer=SurvScorer(sksurv.metrics.cumulative_dynamic_auc, 1, {k: globals_dict[k] for k in scorer_args_keys}),
        )


def get_all_metric_containers(
    globals_dict: dict, raise_errors: bool = True
) -> Dict[str, RegressionMetricContainer]:
    return pycaret.containers.base_container.get_all_containers(
        globals(), globals_dict, RegressionMetricContainer, raise_errors
    )
