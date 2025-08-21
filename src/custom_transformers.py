# Remove added prefixes in df for readability
def remove_prefix(df):
    X = df.copy()
    X.columns = X.columns.str.removeprefix("num__")  # Only ml has num__ prefix
    X.columns = X.columns.str.removeprefix("cat__")
    X.columns = X.columns.str.removeprefix("remainder__")
    X.columns = X.columns.str.removeprefix("bin__")  # Only nomo has bin__ prefix
    return X
