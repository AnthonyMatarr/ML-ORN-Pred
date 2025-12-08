from src.config import SEED, BASE_PATH

from operator import xor
import warnings
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from MLstatkit import Bootstrapping
import pandas as pd


########################################## Helper Functions ##########################################
def get_percentile_range_thresholds(y_proba, n_bins=4, lower=1, upper=99):
    """
    Returns thresholds using the [lower, upper] percentiles of y_proba, dividing the interval into (as close as possible) equal-width bins.
    """
    # Compute robust min/max using percentiles
    lo = np.percentile(y_proba, lower)
    hi = np.percentile(y_proba, upper)
    edges = np.linspace(lo, hi, n_bins + 1)
    # Internal edges only (drop low/high)
    thresholds = edges[1:-1]
    return thresholds  # 1D array, length n_bins-1


def get_logspace_thresholds(y_proba, n_bins=4, lower=1e-5, upper=None):
    """
    Returns thresholds spaced evenly on a log scale within [lower, upper], gracefully handling zero predictions.
    """
    if upper is None:
        upper = np.percentile(y_proba, 99)  # or use max(y_proba)
    # Avoid log(0) by setting very small lower bound
    lo = max(lower, np.min(y_proba[y_proba > 0]))
    hi = upper
    # Make log-spaced edges
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    thresholds = edges[1:-1]
    return thresholds


def plot_risk_bar_dot(y_true, y_proba, thresholds, ax=None):
    """
    Allocates predictions to bins, calculates outcome rate & mean prediction for each bin,
    and makes a bar graph with risk curve overlay.
    """
    thresholds = np.asarray(thresholds, dtype=float).flatten()
    n_bins = len(thresholds) + 1
    bin_indices = np.digitize(y_proba, thresholds, right=False)  # 0,1,...,n_bins-1

    event_rates = []
    mean_preds = []
    counts = []

    for b in range(n_bins):
        mask = bin_indices == b
        n = mask.sum()
        counts.append(n)
        if n == 0:
            event_rates.append(np.nan)
            mean_preds.append(np.nan)
        else:
            event_rates.append(y_true[mask].mean())
            mean_preds.append(y_proba[mask].mean())

    # Label bins as "Bin 0", "Bin 1", ...
    bins_labels = [f"Bin {i}" for i in range(n_bins)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_bins), event_rates, color="C0", alpha=0.7, label="Event Rate")
    ax.plot(range(n_bins), mean_preds, "o-", color="C1", label="Avg. Predicted Risk")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bins_labels, rotation=0)
    ax.set_ylabel("Fraction With Outcome / Mean Prediction")
    ax.set_xlabel("Risk Bin")
    ax.legend()

    # n=XXXX at bottom of bar
    for i, n in enumerate(counts):
        ax.text(
            i,
            0.0,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="k",
        )
    plt.tight_layout()
    return ax


def get_cm(
    model_name,
    outcome_name,
    data_type,
    y_true,
    y_pred,
    show_output=False,
    results_path=None,
):
    """
    Generate confusion matrix and (optionally) export it
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title(f"{model_name}: {data_type}")
    if results_path:
        cm_path = (
            results_path
            / "figures"
            / "CM"
            / outcome_name
            / f"{model_name}_{data_type}_CM.pdf"
        )
        if cm_path.exists():
            warnings.warn(f"Over-writing confusion matrix at path: {cm_path}")
            cm_path.unlink()
        cm_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(cm_path, bbox_inches="tight")
    if show_output:
        plt.show()
    else:
        plt.close()


def get_discrimination_str(
    *_,
    y_true,
    y_proba,
    metric_str,
    threshold,
    n_bootstraps=5000,
    random_state=SEED,
    bin_thresholds=None,
    show_progress=False,
):
    """
    Calculate a classification metric with 95% confidence interval using bootstrap resampling.

    Computes a performance metric (e.g., AUROC, F1, accuracy) and its bootstrapped
    confidence interval, returning a formatted string for reporting.

    Parameters
    ----------
    *_ : tuple
        Placeholder to prevent positional arguments (raises ValueError if used).
    y_true : np.ndarray
        True binary class labels (0 or 1).
    y_proba : np.ndarray
        Predicted probabilities for the positive class (continuous values in [0, 1]).
    metric_str : str
        Metric to calculate. Must be one of: 'f1', 'accuracy', 'recall', 'precision',
        'roc_auc', 'average_precision', 'pr_auc', 'ici', 'brier'.
    threshold : float
        Probability threshold for converting predictions to hard labels.
        Used for threshold-based metrics (F1, accuracy, recall, precision).
    n_bootstraps : int, default=5000
        Number of bootstrap iterations for confidence interval estimation.
    random_state : int, default=SEED
        Random seed for reproducibility of bootstrap sampling.
    bin_thresholds : array-like, optional
        Bin edges for calibration metrics (e.g., ICI). Only used when metric_str='ici'.
    show_progress : bool, default=False
        If True, displays a progress bar during bootstrap iterations.

    Returns
    -------
    str
        Formatted string: '<metric_val> (<ci_lower>, <ci_upper>)' with values
        rounded to 3 decimal places.

    Raises
    ------
    ValueError
        If positional arguments are provided.
    ValueError
        If metric_str is not one of the accepted values.

    Notes
    -----
    - Uses MLStakit.Bootstrapping for percentile-based confidence intervals
    - Confidence level is fixed at 95%
    - Threshold-dependent metrics (F1, accuracy, recall, precision) use the
      specified threshold to convert probabilities to binary predictions
    - Calibration metrics (ICI) require bin_thresholds to be specified
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if metric_str == "ici":
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            bin_thresholds=bin_thresholds,
            show_progress=show_progress,
        )
    else:
        metric_val, ci_lower, ci_upper = Bootstrapping(
            y_true,
            y_proba,
            metric_str=metric_str,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            threshold=threshold,
            random_state=random_state,
            show_progress=show_progress,
        )
    final_str = f"{metric_val:.3f} ({ci_lower:.3f}, {ci_upper:.3f})"
    return final_str


def plot_ROC(
    y_true, y_proba, data_type, n_bootstraps=5000, seed=SEED, show_progress=False
):
    """
    Plot ROC curve and calculate AUROC with confidence interval and optimal threshold.

    Generates an ROC curve plot on the current matplotlib axes and computes the
    area under the curve with bootstrapped 95% confidence intervals. Also determines
    the optimal probability threshold using Youden's J statistic.

    Parameters
    ----------
    y_true : array-like
        True binary class labels (0 or 1).
    y_proba : array-like
        Predicted probabilities for the positive class (continuous values in [0, 1]).
    data_type : str
        Label for the dataset (e.g., 'Train', 'Test', 'Validation') used in the
        plot legend.
    n_bootstraps : int, default=5000
        Number of bootstrap iterations for confidence interval estimation.
    seed : int, default=SEED
        Random seed for reproducibility of bootstrap sampling.
    show_progress : bool, default=False
        If True, displays a progress bar during bootstrap iterations.

    Returns
    -------
    auc_string : str
        Formatted AUROC with confidence interval: '<auc> (<ci_lower>-<ci_upper>)'
        with values rounded to 3 decimal places.
    optimal_threshold : float
        Optimal probability threshold that maximizes Youden's J statistic
        (sensitivity + specificity - 1). This threshold balances true positive
        and false positive rates.

    Notes
    -----
    - Adds a line to the current matplotlib axes; does not create a new figure
    - Uses Youden's J statistic (max(TPR - FPR)) to identify optimal threshold
    - Confidence intervals are computed using MLStakit.Bootstrapping at 95% level
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc, lower_CI, upper_CI = Bootstrapping(
        y_true,
        y_proba,
        random_state=seed,
        metric_str="roc_auc",
        n_bootstraps=n_bootstraps,
        show_progress=show_progress,
    )
    auc_string = f"{auc:.3f} ({lower_CI:.3f}-{upper_CI:.3f})"
    model_score = f"AUROC = {auc_string}"

    ## Youden's J to determine threshold ##
    pr_dif = tpr - fpr
    optimal_idx = np.argmax(pr_dif)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr, tpr, lw=4, label=f"{data_type} {model_score}")
    return auc_string, optimal_threshold


########################################## Main function ##########################################
def evaluate_models(
    *_,
    model_dict,
    outcome_name="ORN",
    data_dict,
    n_bins=4,
    results_path=None,
    n_bootstraps=5000,
    show_cm=False,
    show_roc=False,
    show_cal=False,
    show_progress=False,
):
    """
    Comprehensive evaluation of binary classification models across train/val/test sets.

    For each model, this function:
    1. Generates predicted probabilities and plots ROC curves
    2. Calculates AUROC with bootstrapped confidence intervals
    3. Determines optimal prediction threshold using Youden's J statistic
    4. Computes discrimination metrics (F1, accuracy, recall, precision, Brier, ICI)
    5. Creates calibration plots and risk stratification visualizations
    6. Generates confusion matrices
    7. Exports results, plots, and predictions for downstream use

    Parameters
    ----------
    *_ : tuple
        Placeholder to prevent positional arguments (raises ValueError if used).
    model_dict : dict of {str: sklearn-compatible model}
        Maps model names to trained models.
        Each model must implement predict_proba().
    outcome_name : str, default ORN
        Name of the outcome being predicted. Used for file naming and plot titles.
    ########################################################################################################################
    data_dict -----> NEEDS WORK############################################################
    ##########################################################################################
    n_bins : int, default=4
        Number of risk stratification bins for calibration analysis.
    results_path : pathlib.Path, optional
        Directory path for saving results, figures, and model artifacts.
        If None, results are not saved to disk.
    n_bootstraps : int, default=5000
        Number of bootstrap iterations for confidence interval estimation.
    show_cm : bool, default=False
        If True, displays confusion matrix plots.
    show_roc : bool, default=False
        If True, displays ROC curve plots.
    show_cal : bool, default=False
        If True, displays calibration and risk stratification plots.
    show_progress : bool, default=False
        If True, shows progress bars during bootstrap iterations.

    Returns
    -------
    dict{str: dict{str: dict{str: str or float}}}
        Nested dictionary containing evaluation metrics for each model and dataset.
        Structure:
        {
            'train': {
                <model_name>: {
                    'AUROC (95% CI)': '<auc> (<ci_lower>, <ci_upper>)',
                    'Threshold': <float>,
                    'f1': '<f1> (<ci_lower>, <ci_upper>)',
                    'accuracy': '<acc> (<ci_lower>, <ci_upper>)',
                    'recall': '<recall> (<ci_lower>, <ci_upper>)',
                    'precision': '<prec> (<ci_lower>, <ci_upper>)',
                    'brier': '<brier> (<ci_lower>, <ci_upper>)',
                    'ici': '<ici> (<ci_lower>, <ci_upper>)'
                }
            },
            'val': {...},  # Same structure as 'train'
            'test': {...}  # Same structure, only if X_test/y_test provided
        }

    Raises
    ------
    ValueError
        If positional arguments are provided.
    ValueError
        If only one of X_test or y_test is None (both must be provided or omitted).

    Notes
    -----
    - Threshold is determined using Youden's J statistic (max sensitivity + specificity)
    - Risk bins use log-scale spacing by default for better stratification
    - For models named 'stack', additional artifacts are saved for use in interface:
      * Bin thresholds to BASE_PATH/app/bin_thresholds/{outcome_name}.npz
      * All predictions to BASE_PATH/app/all_preds/{outcome_name}.parquet
    - Figures are organized by type and saved to results_path/figures/{type}/{outcome_name}/
    - All metrics use bootstrapped 95% confidence intervals
    - ICI (Integrated Calibration Index) uses the log-scale bin thresholds
    - If results_path is provided, existing files are overwritten with warnings

    Files Created (if results_path specified)
    ------------------------------------------
    - ROC curves: results_path/figures/ROC/{outcome_name}/{model_name}_ROC.pdf
    - Risk stratification: results_path/figures/risk_bins/{outcome_name}/{model_name}.pdf
    - Confusion matrices: results_path/figures/confusion_matrix/{outcome_name}/{model_name}_{dataset}.pdf
    - Bin thresholds (stack only): BASE_PATH/app/bin_thresholds/{outcome_name}.npz
    - All predictions (stack only): BASE_PATH/app/all_preds/{outcome_name}.parquet
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    CLASS_REPORT_DICT = {"train": {}, "test": {}}
    ## For each model
    for model_name, model in model_dict.items():
        if model_name == "NLR":
            X_train = data_dict["nomo"]["X_train"]
            X_test = data_dict["nomo"]["X_test"]
        else:
            X_train = data_dict["base"]["X_train"]
            X_test = data_dict["base"]["X_test"]
        y_train = data_dict["base"]["y_train"].values.ravel()
        y_test = data_dict["base"]["y_test"].values.ravel()
        print(f"Model: {model_name}...")
        # ================== ADD TO CLASS REPORT ===================
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
        #################################################################################################################
        ########################################### AUROC + binary thresholds ###########################################
        #################################################################################################################
        print(f"\t Dealing with AUROC...")
        # ================== Add to class report ===================
        plt.figure(figsize=(12, 8))
        plt.plot(
            [0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier"
        )
        train_roc_str, train_estimated_threshold = plot_ROC(
            y_train,
            y_proba_train,
            "Training",
            n_bootstraps=n_bootstraps,
            show_progress=show_progress,
        )
        test_roc_str, _ = plot_ROC(
            y_test,
            y_proba_test,
            "Testing",
            n_bootstraps=n_bootstraps,
            show_progress=show_progress,
        )
        # ================== ADD TO CLASS REPORT ===================
        print(f"\t Threshold determined by train set")
        binary_threshold = train_estimated_threshold
        # ================== Add to class report ===================
        CLASS_REPORT_DICT["train"][model_name] = {
            "AUROC (95% CI)": train_roc_str,
            "Threshold": round(binary_threshold, 3),
        }
        CLASS_REPORT_DICT["test"][model_name] = {
            "AUROC (95% CI)": test_roc_str,
            "Threshold": round(binary_threshold, 3),
        }
        # ================== PLOT ===================
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=21, fontweight=550)
        plt.ylabel("True Positive Rate", fontsize=21, fontweight=550)
        plt.tick_params(axis="both", which="major", labelsize=15)
        plt.title(f"{model_name} ROC", fontweight="semibold", fontsize=25)
        plt.legend(loc="lower right", prop={"size": 19, "weight": 550})
        if results_path:
            roc_path = (
                Path(results_path)
                / "figures"
                / "ROC"
                / outcome_name
                / f"{model_name}_ROC.pdf"
            )
            if roc_path.exists():
                warnings.warn(f"Over-writing roc-curve at path {roc_path}")
                roc_path.unlink()
            roc_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(roc_path, bbox_inches="tight")
        if show_roc:
            plt.show()
        else:
            plt.close()
        #################################################################################################################
        ############################################## Risk Bins ########################################################
        #################################################################################################################
        # ================== GET BIN THRESHOLDS ===================
        # Use train + val set
        # train_val_probs = np.concatenate([y_proba_train, y_proba_val])
        train_val_probs = y_proba_train
        ## linear-scale ##
        # thresholds = get_percentile_range_thresholds(train_val_probs, n_bins=n_bins)
        ##################
        ## log-scale ##
        bin_thresholds = get_logspace_thresholds(train_val_probs, n_bins=n_bins)
        ################
        # ================== PLOT RISK BARS ===================
        ax = plot_risk_bar_dot(y_test, y_proba_test, bin_thresholds)
        plt.title(f"{model_name} Test Risk Stratification: {outcome_name}")
        if results_path:
            bin_plot_path = (
                results_path
                / "figures"
                / "risk_bins"
                / outcome_name
                / f"{model_name}.pdf"
            )
            if bin_plot_path.exists():
                print(f"Over-writing bin plot at path {bin_plot_path}")
            bin_plot_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(bin_plot_path, bbox_inches="tight")
        if show_cal:
            plt.show()
        else:
            plt.close()
        #################################################################################################################
        ########################################### Get discrimination metrics ##########################################
        #################################################################################################################
        print(f"\t Getting discrimination metrics...")
        # ================== Get predictions ===================
        y_pred_train = (y_proba_train >= binary_threshold).astype(int)
        y_pred_test = (y_proba_test >= binary_threshold).astype(int)  # type: ignore
        # ================== Confusion Matrices ===================
        ##Train
        get_cm(
            model_name,
            outcome_name,
            "Train",
            y_train,
            y_pred_train,
            show_cm,
            results_path=results_path,
        )
        get_cm(
            model_name,
            outcome_name,
            "Test",
            y_test,
            y_pred_test,
            show_cm,
            results_path=results_path,
        )
        # ================== Get accuracy, recall, precision, brier, ici ===================
        metrics_strs = ["f1", "accuracy", "recall", "precision", "brier", "ici"]
        for metric_str in metrics_strs:
            ## Only need bin thresholds for ICI
            if metric_str == "ici":
                bin_thresholds_for_ici = bin_thresholds
            else:
                bin_thresholds_for_ici = None
            ## Train
            CLASS_REPORT_DICT["train"][model_name][metric_str] = get_discrimination_str(
                y_true=y_train,
                y_proba=y_proba_train,
                metric_str=metric_str,
                threshold=binary_threshold,
                n_bootstraps=n_bootstraps,
                random_state=SEED,
                bin_thresholds=bin_thresholds_for_ici,
                show_progress=show_progress,
            )
            ##Test
            CLASS_REPORT_DICT["test"][model_name][metric_str] = get_discrimination_str(
                y_true=y_test,
                y_proba=y_proba_test,
                metric_str=metric_str,
                threshold=binary_threshold,
                n_bootstraps=n_bootstraps,
                random_state=SEED,
                bin_thresholds=bin_thresholds_for_ici,
                show_progress=show_progress,
            )
    return CLASS_REPORT_DICT
