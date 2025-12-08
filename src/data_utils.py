from src.config import BASE_PATH
import joblib
import pandas as pd


def get_data(is_nomo, file_dir=BASE_PATH / "data"):
    """
    For a given outcome, get X/y train, validation, and testing data
    """
    if is_nomo:
        data_dict = {
            "X_train": pd.read_parquet(
                file_dir / "processed" / "nomo_train_transformed.parquet"
            ),
            "y_train": pd.read_excel(
                file_dir / "raw" / "split" / "Raw_y_train.xlsx", index_col=0
            ),
            "X_test": pd.read_parquet(
                file_dir / "processed" / "nomo_test_transformed.parquet"
            ),
            "y_test": pd.read_excel(
                file_dir / "raw" / "split" / "Raw_y_test.xlsx", index_col=0
            ),
        }
    else:
        data_dict = {
            "X_train": pd.read_parquet(
                file_dir / "processed" / "ml_train_transformed.parquet"
            ),
            "y_train": pd.read_excel(
                file_dir / "raw" / "split" / "Raw_y_train.xlsx", index_col=0
            ),
            "X_test": pd.read_parquet(
                file_dir / "processed" / "ml_test_transformed.parquet"
            ),
            "y_test": pd.read_excel(
                file_dir / "raw" / "split" / "Raw_y_test.xlsx", index_col=0
            ),
        }
    return data_dict


def get_models(model_prefix_list, file_dir=BASE_PATH / "v1.0.0_legacy" / "models"):
    """
    For a given outcome, get all models that predict that outcome
    """
    model_dict = {}
    for model_name in model_prefix_list:
        model = joblib.load(file_dir / f"{model_name}.joblib")
        model_dict[model_name] = model
    return model_dict
