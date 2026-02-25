"""Project constants and configuration."""

settings = {
    "cols_to_unify": [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ],
    "binary_cols": [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
    ],
    "cols_to_drop": [
        "StreamingTV",
        "PhoneService",
        "MultipleLines",
        "StreamingMovies",
        "DeviceProtection",
        "Partner",
        "gender",
    ],
}
