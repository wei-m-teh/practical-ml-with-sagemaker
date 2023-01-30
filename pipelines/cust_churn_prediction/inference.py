import json
import os
from xgboost import XGBClassifier
import numpy as np
from io import StringIO
import pandas as pd

import sagemaker_xgboost_container.encoder as xgb_encoders

def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model_file = "xgboost-churn-model.json"
    booster = XGBClassifier()
    booster.load_model(fname=os.path.join(model_dir, model_file))
    return booster


def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.

    Return a DMatrix (an object that can be passed to predict_fn).
    """
    if request_content_type == "text/csv":
        print(f"request_body: {request_body}")
        return pd.read_csv(StringIO(request_body), sep=",", header=None)
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )


def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    prediction = model.predict(input_data)
    return prediction