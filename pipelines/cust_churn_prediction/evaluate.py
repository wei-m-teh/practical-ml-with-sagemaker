import json
import pathlib
import tarfile
import numpy as np
import pandas as pd
import xgboost
import datetime as dt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import os
import glob
from xgboost import XGBClassifier

if __name__ == "__main__":   
    #Read Model Tar File
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model_file = "xgboost-churn-model.json"
    model = XGBClassifier()
    model.load_model(fname=model_file)
    #Read Test Data using which we evaluate the model
    
    all_test_files = glob.glob(os.path.join('/opt/ml/processing/validation', "*.csv"))
    df = pd.concat((pd.read_csv(f, header=None) for f in all_test_files), ignore_index=True)

    y_test = df.iloc[:, 1].to_numpy()
    df.drop([0,1], axis=1, inplace=True)
    #Run Predictions
    y_pred = model.predict(df.values)
    y_pred_proba=model.predict_proba(df.values)[:,1]
    #Evaluate Predictions
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = np.round(accuracy_score(y_test,y_pred)*100,2)
    precision = np.round(precision_score(y_test,y_pred)*100,2)
    recall = np.round(recall_score(y_test,y_pred)*100,2)
    f1_score = np.round(f1_score(y_test,y_pred)*100,2)

    report_dict = {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
            "recall" : {
                "value" : recall
            },
            "precision" : {
                "value" : precision
            },
            "accuracy" : {
                "value" : accuracy
            },
            "f1" : {
                "value" : f1_score
            }
        },
    }
    #Save Evaluation Report
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
