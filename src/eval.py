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
    Generate bin thresholds using linear spacing between specified percentiles of predicted probabilities.
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
    Generate bin thresholds using logarithmic spacing, suitable for risk stratification with skewed predictions.
    """
    if upper is None:
        upper = np.percentile(y_proba, 99)  # or can use max(y_proba)
    # avoid log(0) by setting very small lower bound
    lo = max(lower, np.min(y_proba[y_proba > 0]))
    hi = upper
    # Make log-spaced edges
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    thresholds = edges[1:-1]
    return thresholds


def plot_risk_bar_dot(y_true, y_proba, thresholds, ax=None):
    """
    Create risk stratification plot with bar graph showing observed event rates and overlaid mean predictions per bin.
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

    ## Label bins w/ thresholds
    bins_labels = []
    for i in range(n_bins):
        if i == 0:
            bins_labels.append(f"Bin {i}\n[0, {thresholds[0]:.2f})")
        elif i == n_bins - 1:
            bins_labels.append(f"Bin {i}\n[{thresholds[-1]:.2f}, 1]")
        else:
            bins_labels.append(f"Bin {i}\n[{thresholds[i-1]:.2f}, {thresholds[i]:.2f})")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Left axis (ax) for bars
    bar_color = "C0"
    ax.bar(range(n_bins), event_rates, color=bar_color, alpha=0.7, label="ORN Rate")
    ax.set_ylabel("ORN Rate", color=bar_color)
    ax.tick_params(axis="y", labelcolor=bar_color)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Right axis (ax2) for line plot
    ax2 = ax.twinx()
    line_color = "C1"
    ax2.plot(
        range(n_bins), mean_preds, "o-", color=line_color, label="Avg. Predicted Risk"
    )
    ax2.set_ylabel("Mean Model Prediction", color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))

    # X-axis settings (shared)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bins_labels, rotation=0)
    ax.set_xlabel("Risk Bin")
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

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
    return ax, ax2


def get_cm(
    model_name,
    data_type,
    y_true,
    y_pred,
    show_output=False,
    results_path=None,
):
    """
    Generate and optionally save confusion matrix heatmap for binary classification predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title(f"{model_name}: {data_type}")
    if results_path:
        cm_path = results_path / "figures" / "CM" / f"{model_name}_{data_type}_CM.pdf"
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
    Calculate discrimination metric with bootstrapped 95% CI, returning formatted string.

    Supports: f1, accuracy, recall, precision, roc_auc, brier, and ICI (integrated calibration index).
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
    Plot ROC curve with bootstrapped AUROC CI and determine optimal classification threshold using Youden's J.
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
    X_train,
    y_train,
    X_val,
    y_val,
    X_test=None,
    y_test=None,
    n_bins=4,
    results_path=None,
    threshold_str="val",
    n_bootstraps=5000,
    show_cm=False,
    show_roc=False,
    show_cal=False,
    show_progress=False,
):
    """
    Comprehensive model evaluation: ROC/AUROC, optimal thresholds, discrimination metrics, risk stratification, and calibration.

    Returns nested dict with metrics for train/val/test splits across all models.
    """
    if _ != tuple():
        raise ValueError("This function does not take positional arguments")
    if xor(X_test is None, y_test is None):
        raise ValueError(
            "One of X_test or y_test is None while the other is not. The presence of these arguments much match!"
        )
    CLASS_REPORT_DICT = {"train": {}, "val": {}, "test": {}}
    ## For each model
    for model_name, model in model_dict.items():
        print(f"Model: {model_name}...")
        # ================== ADD TO CLASS REPORT ===================
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_val = model.predict_proba(X_val)[:, 1]
        if X_test is not None:
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
        val_roc_str, val_estimated_threshold = plot_ROC(
            y_val,
            y_proba_val,
            "Validation",
            n_bootstraps=n_bootstraps,
            show_progress=show_progress,
        )
        if X_test is not None:
            test_roc_str, _ = plot_ROC(
                y_test,
                y_proba_test,
                "Testing",
                n_bootstraps=n_bootstraps,
                show_progress=show_progress,
            )
        # ================== ADD TO CLASS REPORT ===================
        if threshold_str == "val":
            print(f"\t Threshold determined by validation set")
            binary_threshold = val_estimated_threshold
        elif threshold_str == "train":
            print(f"\t Threshold determined by train set")
            binary_threshold = train_estimated_threshold
        else:
            warnings.warn(
                f'Invalid input to "<threshold_str>" provided. Needs to be one of {"val", "train"}, got {threshold_str} instead. Using 0.5 as a threshold.'
            )
            binary_threshold = 0.5
        # ================== Add to class report ===================
        CLASS_REPORT_DICT["train"][model_name] = {
            "AUROC (95% CI)": train_roc_str,
            "Threshold": round(binary_threshold, 3),
        }
        CLASS_REPORT_DICT["val"][model_name] = {
            "AUROC (95% CI)": val_roc_str,
            "Threshold": round(binary_threshold, 3),
        }
        if X_test is not None:
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
            roc_path = Path(results_path) / "figures" / "ROC" / f"{model_name}_ROC.pdf"
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
        train_val_probs = np.concatenate([y_proba_train, y_proba_val])
        ## linear-scale ##
        # thresholds = get_percentile_range_thresholds(train_val_probs, n_bins=n_bins)
        ##################
        ## log-scale ##
        ## Use just the train set
        bin_thresholds = get_logspace_thresholds(train_val_probs, n_bins=n_bins)
        ###############
        if results_path:
            bins_path = results_path / "bin_thresholds" / f"{model_name}.npz"
            if bins_path.exists():
                print(f"Over-writing bin data at path {bins_path}")
                bins_path.unlink()
            bins_path.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                bins_path,
                thresholds=bin_thresholds,
            )
        # ================== PLOT RISK BARS ===================
        if X_test is not None:
            ax = plot_risk_bar_dot(y_test, y_proba_test, bin_thresholds)
            plt.title(f"{model_name} Test Risk Stratification")
            if results_path:
                bin_plot_path = (
                    results_path / "figures" / "risk_bins" / f"{model_name}.pdf"
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
        ################################### All predictions (for interface) #############################################
        #################################################################################################################
        # ## ONLY compute if export is desired
        # if results_path:
        #     # file path
        #     all_pred_path = BASE_PATH / "app" / "all_preds.parquet"
        #     if all_pred_path.exists():
        #         print(f"Over-writing all preds at path {all_pred_path}")
        #         all_pred_path.unlink()
        #     all_pred_path.parent.mkdir(exist_ok=True, parents=True)
        #     # get preds
        #     all_probs = np.concatenate([y_proba_train, y_proba_val, y_proba_test])
        #     all_labels = np.concatenate([y_train, y_val, y_test])  # type: ignore
        #     all_predictions = pd.DataFrame({"prob": all_probs, "label": all_labels})
        #     all_predictions.to_parquet(all_pred_path)
        #################################################################################################################
        ########################################### Get discrimination metrics ##########################################
        #################################################################################################################
        print(f"\t Getting discrimination metrics...")
        # ================== Get predictions ===================
        y_pred_train = (y_proba_train >= binary_threshold).astype(int)
        y_pred_val = (y_proba_val >= binary_threshold).astype(int)
        if X_test is not None:
            y_pred_test = (y_proba_test >= binary_threshold).astype(int)  # type: ignore
        # ================== Confusion Matrices ===================
        ##Train
        get_cm(
            model_name,
            "Train",
            y_train,
            y_pred_train,
            show_cm,
            results_path=results_path,
        )
        ## Val
        get_cm(
            model_name,
            "Validation",
            y_val,
            y_pred_val,
            show_cm,
            results_path=results_path,
        )
        if X_test is not None:
            get_cm(
                model_name,
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
            ##Val
            CLASS_REPORT_DICT["val"][model_name][metric_str] = get_discrimination_str(
                y_true=y_val,
                y_proba=y_proba_val,
                metric_str=metric_str,
                threshold=binary_threshold,
                n_bootstraps=n_bootstraps,
                random_state=SEED,
                bin_thresholds=bin_thresholds_for_ici,
                show_progress=show_progress,
            )
            ##Test
            if X_test is not None:
                CLASS_REPORT_DICT["test"][model_name][metric_str] = (
                    get_discrimination_str(
                        y_true=y_test,
                        y_proba=y_proba_test,
                        metric_str=metric_str,
                        threshold=binary_threshold,
                        n_bootstraps=n_bootstraps,
                        random_state=SEED,
                        bin_thresholds=bin_thresholds_for_ici,
                        show_progress=show_progress,
                    )
                )
    return CLASS_REPORT_DICT
