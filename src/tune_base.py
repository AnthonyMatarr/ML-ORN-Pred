from src.config import SEED

## General purpose
import joblib
import threading
import json
import warnings
import logging
import numpy as np
import optuna
from func_timeout import func_timeout, FunctionTimedOut
from sklearn.exceptions import ConvergenceWarning
import time
from sklearn.model_selection import train_test_split
import torch

## Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from src.nn_model import TorchNNClassifier

warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="Cannot compute class probabilities")
## GLOBALS

MODEL_N_JOBS = {
    "lr": 1,
    "lgbm": 1,
    "xgb": 1,
    "knn": 1,
    "svc": 1,
}
LOG_LOCK = threading.Lock()
STAGE = 1
if STAGE == 1:
    N_SPLITS = 4
elif STAGE == 2:
    N_SPLITS = 5
else:
    raise ValueError("Stage must be one of [1,2]")

N_PARALLEL_CV_JOBS = 1


#############################################################################################
###################################### MODEL BUILDERS #######################################
#############################################################################################
# ========================> LOGISTIC REGRESSION
def lr_model_builder_stage1(trial):
    C = trial.suggest_float("C", 1e-4, 10.0, log=True)

    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

    if penalty == "elasticnet":
        solver = "saga"
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    elif penalty == "l1":
        solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
        l1_ratio = None
    else:  # l2
        solver = trial.suggest_categorical(
            "solver_l2", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
        )
        l1_ratio = None

    # Moderately constrained pos_weight
    if solver in ["saga", "sag"]:
        pos_weight = trial.suggest_float("pos_weight_saga", 1.0, 8.0, log=True)
    elif solver == "liblinear":
        pos_weight = trial.suggest_float("pos_weight_linear", 1.0, 12.0, log=True)
    else:
        pos_weight = trial.suggest_float("pos_weight_general", 1.0, 8.0, log=True)

    class_weight = {0: 1.0, 1: pos_weight}

    if solver == "liblinear":
        intercept_scaling = trial.suggest_float(
            "intercept_scaling", 1e-2, 1e2, log=True
        )
        n_jobs = 1
    else:
        intercept_scaling = 1.0
        n_jobs = MODEL_N_JOBS["lr"]

    max_iter = 5000 if solver in ["saga", "sag"] else 4000

    return LogisticRegression(
        penalty=penalty,
        C=C,
        tol=1e-4,
        fit_intercept=True,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=SEED,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        warm_start=False,
        n_jobs=n_jobs,
    )


def lr_model_builder_stage2(trial):
    pass


# ========================> LIGHTGBM
def lightgbm_model_builder_stage1(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 200, 1200)

    max_depth = trial.suggest_int("max_depth", 3, 12)
    max_leaves = min(512, 2**max_depth)
    min_leaves = min(16, max_leaves)
    num_leaves = trial.suggest_int("num_leaves", min_leaves, max_leaves)

    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 20, 200)
    min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.0, 3.0)

    feature_fraction = trial.suggest_float("feature_fraction", 0.6, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 10)

    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 15.0, log=True)

    pos_bagging_fraction = trial.suggest_float("pos_bagging_fraction", 0.7, 1.0)
    neg_bagging_fraction = trial.suggest_float("neg_bagging_fraction", 0.2, 1.0)

    lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 20.0)
    lambda_l2 = trial.suggest_float("lambda_l2", 0.0, 30.0)

    max_bin = trial.suggest_int("max_bin", 64, 300)

    return LGBMClassifier(
        objective="binary",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        min_split_gain=min_gain_to_split,
        feature_fraction=feature_fraction,
        pos_bagging_fraction=pos_bagging_fraction,
        neg_bagging_fraction=neg_bagging_fraction,
        bagging_freq=bagging_freq,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        scale_pos_weight=scale_pos_weight,
        max_bin=max_bin,
        tree_learner="feature_parallel",
        n_jobs=MODEL_N_JOBS["lgbm"],
        seed=SEED,
        bagging_seed=SEED,
        feature_fraction_seed=SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


def lightgbm_model_builder_stage2(trial):
    pass


# ========================> XGBOOST
def xgb_model_builder_stage1(trial):
    sampling_strategy = trial.params.get("sampling_strategy", "none")

    if sampling_strategy == "tomek":
        n_estimators = trial.suggest_int("n_estimators", 200, 700)
    else:
        n_estimators = trial.suggest_int("n_estimators", 300, 1200)

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 8)

    gamma = trial.suggest_float("gamma", 0.0, 5.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 20.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 25.0)

    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    colsample_bylevel = 0.85  # keep fixed in stage 1

    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)

    min_child_weight = trial.suggest_int("min_child_weight", 1, 20)

    return XGBClassifier(
        objective="binary:logistic",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=min_child_weight,
        tree_method="hist",
        n_jobs=MODEL_N_JOBS["xgb"],
        random_state=SEED,
        eval_metric="auc",
    )


def xgb_model_builder_stage2(trial):
    pass


# ========================> K-NEAREST NEIGHBORS
def knn_model_builder_stage1(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 3, 50)

    weights = trial.suggest_categorical("weights", ["uniform", "distance"])

    metric = trial.suggest_categorical(
        "metric", ["euclidean", "manhattan", "minkowski"]
    )

    if metric == "minkowski":
        p = trial.suggest_int("p", 1, 5)
    else:
        p = 2  # Default for euclidean/manhattan

    algorithm = trial.suggest_categorical(
        "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
    )

    # Leaf size only matters for ball_tree and kd_tree
    if algorithm in ["ball_tree", "kd_tree", "auto"]:
        leaf_size = trial.suggest_int("leaf_size", 10, 50)
    else:
        leaf_size = 30  # Default

    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        n_jobs=MODEL_N_JOBS["knn"],
    )


def knn_model_builder_stage2(trial):
    pass


# ========================> SUPPORT VECTOR CLASSIFIER
def svc_model_builder_stage1(trial):
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)

    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])

    if kernel == "rbf":
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    elif kernel == "poly":
        gamma = trial.suggest_categorical("gamma_poly", ["scale", "auto"])
        degree = trial.suggest_int("degree", 2, 4)
        coef0 = trial.suggest_float("coef0", 0.0, 10.0)
    else:  # linear
        gamma = "scale"
        degree = 3
        coef0 = 0.0

    class_weight_mode = trial.suggest_categorical(
        "class_weight_mode", ["balanced", "custom"]
    )

    if class_weight_mode == "custom":
        pos_weight = trial.suggest_float("pos_weight", 1.0, 20.0, log=True)
        class_weight = {0: 1.0, 1: pos_weight}
    else:
        class_weight = "balanced"

    return SVC(
        C=C,
        kernel=kernel,
        degree=degree if kernel == "poly" else 3,
        gamma=gamma,
        coef0=coef0 if kernel == "poly" else 0.0,
        class_weight=class_weight,
        probability=True,
        random_state=SEED,
        max_iter=3000,
        cache_size=500,  # MB of cache for kernel computations
    )


def svc_model_builder_stage2(trial):
    pass


# ========================> NEURAL NETWORK
def nn_model_builder_stage1(trial):
    """
    Neural network model builder for Stage 1 - optimized for small dataset (300 samples).

    Key considerations:
    - Simpler architectures (1-2 layers) to prevent overfitting
    - Smaller hidden layer sizes appropriate for sample size
    - Higher dropout for regularization
    - Strong weight decay
    - Moderate epochs with early stopping recommended
    """
    # Architecture: 1-2 layers only for small dataset
    n_layers = trial.suggest_int("n_layers", 1, 2)

    ### Hidden Layer Sizes ###
    # First layer: moderate size
    hl_1 = trial.suggest_int("hl_1", 16, 64)
    hidden_size_list = [hl_1]

    # Second layer if needed: smaller than first
    if n_layers == 2:
        hl_2 = trial.suggest_int("hl_2", 8, 32)
        hidden_size_list.append(hl_2)

    ### Dropouts - higher for small dataset regularization ###
    dr_1 = trial.suggest_float("dr_1", 0.2, 0.6)
    dropouts = [dr_1]

    if n_layers == 2:
        dr_2 = trial.suggest_float("dr_2", 0.2, 0.6)
        dropouts.append(dr_2)

    ### Activation Function ###
    activation_name = trial.suggest_categorical("act_func_str", ["relu", "leaky_relu"])

    ### Training Hyperparameters ###
    # Learning rate: moderate range
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)

    # Weight decay: strong regularization for small dataset
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    # Optimizer: AdamW better for small data
    optimizer_str = "adamw"

    # Epochs: moderate range (early stopping will handle this)
    num_epochs = trial.suggest_int("num_epochs", 30, 60)

    # Batch size: small for 300 samples, CPU-friendly
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Weight initialization
    weight_init_scheme = trial.suggest_categorical(
        "weight_init_scheme", ["xavier_uniform", "kaiming_uniform"]
    )

    # Bias initialization
    bias_init = 0.0  # Keep at 0 for binary classification

    return TorchNNClassifier(
        hidden_size_list=hidden_size_list,
        dropouts=dropouts,
        activation_name=activation_name,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_str=optimizer_str,
        epochs=num_epochs,
        batch_size=batch_size,
        weight_init_scheme=weight_init_scheme,
        bias_init=bias_init,
        device="cpu",
        verbose=0,
        seed=SEED,
    )


def get_default_params(model_abrv):
    """
    Returns sensible default hyperparameters for each model type.
    Used as fallback if all Optuna trials are pruned.

    Parameters
    ----------
    model_abrv: str
        Model abbreviation: 'lr', 'lgbm', 'xgb','cat' or 'svc'

    Returns
    -------
    dict
        Dictionary of default hyperparameters for the model
    """
    if model_abrv == "lr":
        fallback_model = LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
            max_iter=5000,
            n_jobs=MODEL_N_JOBS["lr"],
        )
        fallback_params = {
            "C": 1.0,
            "penalty": "l2",
            "class_weight": "balanced",
            "solver": "lbfgs",
        }
    elif model_abrv == "lgbm":
        fallback_model = LGBMClassifier(
            objective="binary",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=-1,
            num_leaves=31,
            min_data_in_leaf=20,
            class_weight="balanced",
            n_jobs=MODEL_N_JOBS["lgbm"],
            seed=SEED,
            deterministic=True,
            force_row_wise=True,
            verbosity=-1,
        )
        fallback_params = {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": -1,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
        }
    elif model_abrv == "xgb":
        fallback_model = XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=10.0,  # ~inverse of your 4% positive rate
            tree_method="hist",
            n_jobs=MODEL_N_JOBS["xgb"],
            random_state=SEED,
        )
        fallback_params = {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "scale_pos_weight": 10.0,
        }
    elif model_abrv == "knn":
        fallback_model = KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",  # Distance weighting helps with imbalance
            algorithm="auto",
            metric="euclidean",
            n_jobs=MODEL_N_JOBS["knn"],
        )
        fallback_params = {
            "n_neighbors": 15,
            "weights": "distance",
            "algorithm": "auto",
            "metric": "euclidean",
        }
    elif model_abrv == "svc":
        fallback_model = SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=SEED,
            max_iter=3000,
            cache_size=500,
        )
        fallback_params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": "balanced",
        }
    elif model_abrv == "nn":
        fallback_model = TorchNNClassifier(
            hidden_size_list=[32, 16],
            dropouts=[0.4, 0.4],
            activation_name="relu",
            lr=1e-3,
            weight_decay=1e-3,  # type: ignore
            optimizer_str="adamw",
            epochs=40,
            batch_size=16,
            weight_init_scheme="xavier_uniform",
            bias_init=0.0,
            device="cpu",
            verbose=0,
            seed=SEED,
        )
        fallback_params = {
            "n_layers": 2,
            "hl_1": 32,
            "hl_2": 16,
            "dr_1": 0.4,
            "dr_2": 0.4,
            "act_func_str": "relu",
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "num_epochs": 40,
            "batch_size": 16,
            "weight_init_scheme": "xavier_uniform",
        }

    else:
        raise ValueError(f"Unknown model: {model_abrv}")
    return fallback_model, fallback_params


#############################################################################################
########################################## TUNING ###########################################
#############################################################################################
def make_objective(X_train, y_train, model_builder, outcome_name, scoring="roc_auc"):
    """
    Creates objective for model tuning
    Includes sampling param and calls appropriate model builder
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def objective(trial):
        # build model
        model = model_builder(trial)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=skf,
            n_jobs=N_PARALLEL_CV_JOBS,
        )
        return np.round(np.mean(scores), 4)

    return objective


def tune_models(
    *_,
    model_builder,
    model_abrv,
    outcome_data,
    scoring,
    log_file_path,
    save_path,
    n_parallel_trials,
    n_trials,
    timeout_per_trial=6000,
    clear_progress=False,
):
    """
    Tunes a given model for each given outcome using Optuna and writes params/tuning results to memory.
    Used for LR, LGBM, XGB, KNN, SVC

    Parameters
    ----------
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator
    model_abrv: string
        Abbreviation of model to be tuned, one of ['lr', 'lgbm', 'xgb', 'knn', 'svc']
    outcome_data: dict
        Dictionary containing 'X_train' and 'y_train' keys with training data
    scoring: str
        String specifying which scoring metric to use for tuning.
        Ultimately passed into sklearn.model_selection.cross_val_score().
    log_file_path: pathlib.Path
        Absolute path to file where tuning logs will be written to.
    save_path: pathlib.Path
        Absolute path to directory where best CV score/params are written to in json format.
    n_parallel_trials: int
        Number of parallel Optuna trials to run simultaneously
    n_trials: int
        Total number of Optuna trials to run
    timeout_per_trial: Optional int; defaults to 6000
        Specify max number of seconds a trial can take before abandoning/pruning
        Set to 100 minutes, so essentially timeout pruning is off by default
    clear_progress: Boolean
        If True, check if there's any study progress in db file and delete
        If False, don't check (and therefore maintain any progress if exists)

    Returns
    -------
    dict
        Dictionary with 'best_score' and 'best_params' keys

    Raises
    ------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This func does not take positional args")

    ############ Set up paths ############
    ## Clear log files
    if log_file_path.exists():
        warnings.warn(f"Over-writing log at path: {log_file_path}")
        log_file_path.unlink()
    log_file_path.parent.mkdir(exist_ok=True, parents=True)
    ## Clear JSON path
    if save_path.exists():
        save_path.unlink()
    save_path.parent.mkdir(exist_ok=True, parents=True)

    ## Clear existing studies
    if clear_progress:
        for db_file in save_path.parent.glob(f"{model_abrv}*.db"):
            db_file.unlink()
            with open(log_file_path, "a") as f:
                f.write(f"Deleted existing study database: {db_file.name}\n")

    ############ Set up logger ############
    root_logger = logging.getLogger()  # Root logger
    root_logger.setLevel(logging.INFO)

    # Remove all existing FileHandlers to prevent cross-contamination
    for handler in root_logger.handlers[:]:  # Iterate over copy
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()  # close file handle

    # Now add the handler for current model
    file_handler = logging.FileHandler(log_file_path, mode="a")
    root_logger.addHandler(file_handler)

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    ############ Run for each outcome ############
    result_dict = {}
    with open(log_file_path, "a") as f:
        f.write(f"Starting tuning...\n")
    X_train = outcome_data["X_train"]
    y_train = outcome_data["y_train"].values.ravel()
    #### SUBSAMPLE
    ## Create objective
    base_objective = make_objective(X_train, y_train, model_builder, scoring)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        storage = f"sqlite:///{save_path.parent / f'{model_abrv}.db'}"
        study = optuna.create_study(
            storage=storage,
            study_name=f"{model_abrv}_study",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.HyperbandPruner(),
            load_if_exists=True,
        )

        ###### with timeout ######
        # Helper function for parallel trials with timeout wrapper
        def run_trials(n_trials_per_worker):
            study_local = optuna.load_study(
                study_name=f"{model_abrv}_study",
                storage=storage,
            )

            # Timeout wrapper
            def timeout_objective(trial):
                try:
                    start = time.time()
                    result = func_timeout(
                        timeout_per_trial,
                        base_objective,
                        args=(trial,),
                    )
                    duration = time.time() - start
                    with LOG_LOCK:
                        with open(log_file_path, "a") as f:
                            f.write(
                                f"Trial {trial.number} finished in: {duration:.1f}s \n"
                            )
                    return result
                except FunctionTimedOut:
                    with LOG_LOCK:
                        with open(log_file_path, "a") as f:
                            f.write(
                                f"Trial {trial.number} timed out after {timeout_per_trial}s --> Parameters: {trial.params}\n"
                            )
                    raise optuna.TrialPruned()

            # Timeout_objective instead of base_objective
            study_local.optimize(timeout_objective, n_trials=n_trials_per_worker)  # type: ignore

        # Parallelize trials using threading
        threads = []
        trials_per_worker = int(n_trials / n_parallel_trials)
        for _ in range(n_parallel_trials):
            thread = threading.Thread(target=run_trials, args=(trials_per_worker,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        # Load final results
        study = optuna.load_study(
            study_name=f"{model_abrv}_study",
            storage=storage,
        )
        ####### Optuna timeout safety net #######
        try:
            best_score = study.best_value
            best_params = study.best_params
            ## get results
            result_dict = {
                "best_score": best_score,
                "best_params": best_params,
                # "study": study, # NO need to save this, but including in case want to inspect later
            }

            with open(log_file_path, "a") as f:
                f.write(f"Best {scoring}: {best_score:.4f}\nparams={best_params}\n")
                f.write(f"{'*' * 100}\n")
        except ValueError:
            # All trials were pruned - use default parameters
            with open(log_file_path, "a") as f:
                f.write(
                    f"WARNING: All trials were pruned. "
                    f"Using default hyperparameters.\n"
                )
                # get default model + params
                fallback_model, fallback_params = get_default_params(model_abrv)
                # Run k-fold CV with default params
                skf = StratifiedKFold(
                    n_splits=N_SPLITS, shuffle=True, random_state=SEED
                )
                fallback_scores = cross_val_score(
                    fallback_model,  # type: ignore
                    X_train,
                    y_train,
                    scoring=scoring,
                    cv=skf,
                    n_jobs=N_PARALLEL_CV_JOBS,
                )
                fallback_score = np.mean(fallback_scores)
                result_dict = {
                    "best_score": float(fallback_score),
                    "best_params": fallback_params,
                    "fallback_used": True,  # Used later for building model
                }
                with open(log_file_path, "a") as f:
                    f.write(
                        f"(FALLBACK): {scoring}: {fallback_score:.4f}\n"
                        f"params={fallback_params}\n"
                    )

    # Remove handler when this model is done
    root_logger.removeHandler(file_handler)
    file_handler.close()

    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)

    return result_dict


#############################################################################################
###################################### PRELIM RESULTS #######################################
#############################################################################################
def get_prelim_results(
    *_,
    results_path,
    model_builder,
    model_abrv,
    outcome_data,
    model_save_dir=None,
    cv_splits=5,
):
    """
    Calculates CV mean/std and train AUROC scores and prints them out.
    Also used to export models when model_save_dir is not None.

    Parameters
    ----------
    results_path: pathlib.Path
        Absolute path to json file containing dictionary with 'best_score' and 'best_params'
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator
    model_abrv: str
        Abbreviation of model used: 'lr', 'lgbm', 'xgb', 'knn', 'svc', or 'nn'
    outcome_data: dict
        Dictionary containing 'X_train', 'y_train' keys
    model_save_dir: pathlib.Path; defaults to None
        Directory to save models. If None, will not save models
    cv_splits: int; defaults to 5
        Number of CV folds for final evaluation

    Returns
    --------
    dict: Dictionary with performance metrics

    Raises
    --------
    ValueError:
        If positional arguments are given
    """
    if _ != tuple():
        raise ValueError("This function does not take position arguments!")

    print(f"{'-'*30} {model_abrv} {'-'*30}")
    with open(results_path, "r") as f:
        results = json.load(f)

    ## Get tuning results
    best_score = results["best_score"]
    best_params = results["best_params"]

    # Check if fallback was used
    if results.get("fallback_used", False):
        model, _ = get_default_params(model_abrv)
    else:
        trial = optuna.trial.FixedTrial(best_params)
        model = model_builder(trial)

    ##### Get data #####
    X_train = outcome_data["X_train"]
    y_train = outcome_data["y_train"].values.ravel()

    ##### Perform cross-validation with best params #####
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        model,  # type: ignore
        X_train,
        y_train,
        cv=skf,
        scoring="roc_auc",
        n_jobs=N_PARALLEL_CV_JOBS,
    )

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    ##### Train model on full data #####
    model.fit(X_train, y_train)

    ##### Get train score #####
    train_pred_proba = model.predict_proba(X_train)[:, 1]  # type: ignore
    train_auc = roc_auc_score(y_train, train_pred_proba)

    ##### Export model #####
    if model_save_dir:
        if model_abrv == "nn":
            # Special handling for neural network
            save_path = model_save_dir / f"{model_abrv}.pt"
            if save_path.exists():
                save_path.unlink()
            save_path.parent.mkdir(exist_ok=True, parents=True)

            # Prepare hyperparameters dictionary for saving
            h_params_to_save = {
                "n_layers": best_params.get("n_layers", 1),
                "hl_1": best_params["hl_1"],
                "dr_1": best_params["dr_1"],
                "act_func_str": best_params.get("act_func_str", "relu"),
                "num_epochs": best_params.get("num_epochs", 40),
                "lr": best_params["lr"],
                "weight_decay": best_params["weight_decay"],
                "batch_size": best_params.get("batch_size", 16),
                "weight_init_scheme": best_params.get(
                    "weight_init_scheme", "xavier_uniform"
                ),
            }

            # Add second layer if present
            if best_params.get("n_layers", 1) >= 2:
                h_params_to_save["hl_2"] = best_params["hl_2"]
                h_params_to_save["dr_2"] = best_params["dr_2"]

            # Create checkpoint dictionary
            checkpoint = {
                "h_params": h_params_to_save,
                "state_dict": model.model_.state_dict(),  # type: ignore
                "feature_names_in_": model.feature_names_in_,
            }

            # Save with torch
            torch.save(checkpoint, save_path)
            print(f"Neural network saved to {save_path}")
        else:
            save_path = model_save_dir / f"{model_abrv}.joblib"
            if save_path.exists():
                save_path.unlink()
            save_path.parent.mkdir(exist_ok=True, parents=True)
            joblib.dump(model, save_path)

    ##### Output results #####
    print(f"CV AUROC (mean ± std): \t{cv_mean:.3f} ± {cv_std:.3f}")
    print(f"Train AUROC: \t\t{train_auc:.3f}")
    print(f"PARAMS: \t\t{best_params}")
    print("*" * 10)

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_scores": cv_scores.tolist(),
        "train_auc": train_auc,
        "best_params": best_params,
    }
