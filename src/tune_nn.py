import argparse
from pathlib import Path
import time
import optuna
import pandas as pd
import numpy as np
import logging
import json
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.config import SEED
from src.nn_model import TorchNNClassifier
from sklearn.metrics import roc_auc_score
import warnings


################################################################
##################### Train/Prelim Results #####################
################################################################
def train_and_prelim_eval(data_dict, json_path, model_save_path=None):
    """
    Train neural network with best hyperparameters and evaluate on validation set.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing X_train, y_train, X_val, y_val
    json_path : Path
        Path to JSON file with best hyperparameters
    model_save_path : Path, optional
        Path to save trained model
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"].values.ravel()
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"].values.ravel()

    with open(json_path, "r") as f:
        json_params = json.load(f)

    print(f"Training neural network...")
    print(f"\tBest CV AUROC: \t{json_params['best_score']:.3f}")

    # Extract best parameters
    best_params = json_params["best_params"]

    # Extract architecture parameters
    hidden_sizes = [best_params["hl_1"]]
    dropouts = [best_params["dr_1"]]

    # Add second layer if n_layers >= 2
    if best_params.get("n_layers", 1) >= 2:
        hidden_sizes.append(best_params["hl_2"])
        dropouts.append(best_params["dr_2"])

    # Create and fit model with best params
    clf = TorchNNClassifier(
        hidden_size_list=hidden_sizes,
        dropouts=dropouts,
        activation_name=best_params.get("act_func_str", "relu"),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        optimizer_str="adamw",
        epochs=best_params.get("num_epochs", 80),
        batch_size=best_params.get("batch_size", 16),
        device="cpu",
        seed=SEED,
        # Early stopping with internal validation
        early_stopping=True,
        es_patience=15,
        es_min_delta=0.001,
        val_split=0.20,
        monitor="auc",
        verbose=1,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_proba)

    y_val_proba = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)

    print(f"Train AUROC: \t{train_auc:.3f}")
    print(f"Val AUROC: \t{val_auc:.3f}")
    print(f"Best params: \t{best_params}")
    print("*" * 50)

    # Save model
    if model_save_path:
        h_params_to_save = {
            "n_layers": best_params.get("n_layers", 1),
            "hl_1": best_params["hl_1"],
            "dr_1": best_params["dr_1"],
            "act_func_str": best_params.get("act_func_str", "relu"),
            "num_epochs": best_params.get("num_epochs", 80),
            "lr": best_params["lr"],
            "weight_decay": best_params["weight_decay"],
            "batch_size": best_params.get("batch_size", 16),
        }

        if best_params.get("n_layers", 1) >= 2:
            h_params_to_save["hl_2"] = best_params["hl_2"]
            h_params_to_save["dr_2"] = best_params["dr_2"]

        checkpoint = {
            "h_params": h_params_to_save,
            "state_dict": clf.model_.state_dict(),  # type: ignore
            "feature_names_in_": clf.feature_names_in_,
        }

        if model_save_path.exists():
            warnings.warn(f"Overwriting saved model at {model_save_path}")
            model_save_path.unlink()

        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(checkpoint, model_save_path)
        print(f"Model saved to {model_save_path}")


################################################################
######################## Model Builders ########################
################################################################
def build_nn_estimator_stage1(trial):
    """
    Model builder for stage 1 - optimized for small dataset (300 samples).

    Key changes for small data:
    - Simpler architectures (1-2 layers max)
    - Smaller hidden layer sizes
    - Higher dropout for regularization
    - Smaller batch sizes
    """
    # 1-2 layers only for small dataset
    n_layers = trial.suggest_int("n_layers", 1, 2)

    ### Hidden Layers - smaller for 300 samples ###
    hl_1 = trial.suggest_int("hl_1", 16, 64)
    h_sizes = [hl_1]

    if n_layers == 2:
        hl_2 = trial.suggest_int("hl_2", 8, 32)
        h_sizes.append(hl_2)

    ### Dropouts - higher for regularization ###
    dr_1 = trial.suggest_float("dr_1", 0.2, 0.6)
    dropouts = [dr_1]

    if n_layers == 2:
        dr_2 = trial.suggest_float("dr_2", 0.2, 0.6)
        dropouts.append(dr_2)

    ### Activation ###
    act_name = trial.suggest_categorical("act_func_str", ["relu", "leaky_relu"])

    ### Epochs ###
    num_epochs = trial.suggest_int("num_epochs", 30, 60)

    ### Optimizer ###
    opt_choice = "adamw"
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    ### Batch size ###
    bs = trial.suggest_categorical("batch_size", [8, 16, 32])

    nn_clf = TorchNNClassifier(
        hidden_size_list=h_sizes,
        dropouts=dropouts,
        activation_name=act_name,
        lr=lr,
        weight_decay=wd,
        epochs=num_epochs,
        batch_size=bs,
        optimizer_str=opt_choice,
        device="cpu",
        seed=SEED,
        # Enable early stopping
        early_stopping=True,
        es_patience=10,
        es_min_delta=0.001,
        val_split=0.20,
    )
    return nn_clf


def build_nn_estimator_stage2(trial):
    pass


################################################################
######################### Setup ################################
################################################################
def build_parser():
    parser = argparse.ArgumentParser(
        prog="Neural Network Tuner",
        description="Tune a neural network with Optuna (single outcome, CPU)",
    )
    parser.add_argument("--X_path", required=True, help="Path to feature data (X)")
    parser.add_argument("--y_path", required=True, help="Path to outcome data (y)")
    parser.add_argument(
        "--scoring_str", required=True, help='Scoring metric (e.g., "roc_auc")'
    )
    parser.add_argument(
        "--log_path", required=True, help="Log file path for tuning info"
    )
    parser.add_argument(
        "--results_path",
        required=True,
        help="Output path for best parameters (JSON)",
    )
    parser.add_argument(
        "--n_trials", required=True, type=int, help="Number of Optuna trials"
    )
    parser.add_argument("--seed", required=True, type=int, help="Random seed")
    parser.add_argument(
        "--stage",
        required=True,
        type=int,
        choices=[1, 2],
        help="Tuning stage (1=broad search, 2=refined search)",
    )
    return parser


def load_data(X_path, y_path):
    """Load data from files."""
    X_train = pd.read_parquet(X_path)
    y_train_df = pd.read_excel(y_path, index_col=0)
    y_train = y_train_df.values.ravel()
    return X_train, y_train


def parse_arguments(argv=None):
    return build_parser().parse_args(argv)


def main_tuner(
    *_,
    X_path,
    y_path,
    scoring_str,
    log_path,
    results_path,
    n_trials,
    rand_state,
    stage,
):
    """
    Tune neural network for single outcome on CPU.

    Parameters
    ----------
    stage : int
        Tuning stage (1 or 2)
    """
    if _ != tuple():
        raise ValueError("main_tuner() does not accept positional arguments!")

    # Configure logging
    if log_path.exists():
        log_path.unlink()
    log_path.parent.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(log_path, mode="a")
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    logging.info(f"Starting tuning - Stage {stage}")

    # Set up cross-validation based on stage
    if stage == 1:
        n_splits = 3
        model_builder = build_nn_estimator_stage1
    elif stage == 2:
        n_splits = 5
        model_builder = build_nn_estimator_stage2
    else:
        raise ValueError(f"Stage must be 1 or 2, got {stage}")

    X_train, y_train = load_data(X_path, y_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rand_state)

    def objective(trial):
        """Objective function for Optuna."""
        start_time = time.perf_counter()
        logging.info(f"Starting trial {trial.number}")

        clf = model_builder(trial)

        scores = cross_val_score(
            clf,  # type: ignore
            X_train,
            y_train,
            scoring=scoring_str,
            cv=skf,
            n_jobs=1,
        )

        mean_score = float(np.mean(scores))
        elapsed = time.perf_counter() - start_time
        logging.info(f"Trial {trial.number} finished in: {elapsed:.1f}s \n")

        if trial.number == 0:
            logging.info(
                f"Trial {trial.number} finished with value: {mean_score:.4f} and parameters: {trial.params}. Best is trial {trial.number} with value: {mean_score:.4f}."
            )
        else:
            logging.info(
                f"Trial {trial.number} finished with value: {mean_score:.4f} and parameters: {trial.params}. Best is trial {trial.study.best_trial.number} with value: {trial.study.best_value:.4f}."
            )
        return mean_score

    # Run Optuna study
    study_name = f"nn_stage{stage}_study"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=rand_state),
        pruner=optuna.pruners.HyperbandPruner(min_resource=3, reduction_factor=3),
    )

    logging.info(f"A new study created: {study_name}")
    study.optimize(objective, n_trials=n_trials)

    # Save results
    logging.info(f"Best {scoring_str}: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")

    result = {
        "stage": stage,
        "best_score": round(study.best_value, 4),
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "n_trials": n_trials,
    }

    if results_path.exists():
        results_path.unlink()
    results_path.parent.mkdir(exist_ok=True, parents=True)

    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)

    logging.info(f"Saved best results to {results_path}")

    # Clean up
    root_logger.removeHandler(file_handler)
    file_handler.close()


########################################################
######################### Main #########################
########################################################
def main(argv=None):
    args = parse_arguments(argv)

    # Convert string paths to Path objects
    X_path = Path(args.X_path)
    y_path = Path(args.y_path)
    log_path = Path(args.log_path)
    results_path = Path(args.results_path)

    main_tuner(
        X_path=X_path,
        y_path=y_path,
        scoring_str=args.scoring_str,
        log_path=log_path,
        results_path=results_path,
        n_trials=args.n_trials,
        rand_state=args.seed,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
