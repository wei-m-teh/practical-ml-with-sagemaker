import os
os.system("pip install -U sagemaker smdebug")

import pandas as pd
from xgboost import XGBClassifier
import argparse
import xgboost 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sagemaker.experiments import load_run
from sagemaker.session import Session
import numpy as np
import boto3
import pickle as pkl
import glob
from smdebug import SaveConfig
from smdebug.xgboost import Hook
from time import strftime, gmtime

cols = ['msno', 'is_churn', 'regist_trans', 'mst_frq_plan_days', 'revenue', 'regist_cancels', 'bd', 'tenure', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs', 'city', 'gender', 'registered_via', 'qtr_trans', 'mst_frq_pay_met', 'is_auto_renew']

def create_smdebug_hook(
    out_dir,
    train_data=None,
    validation_data=None,
    frequency=1,
    collections=None,
):

    train_data_csv = "/tmp/train.csv"
    test_data_csv = "/tmp/test.csv"
    
    train_data.iloc[:, 1:].to_csv(train_data_csv, header=False, index=False)
    validation_data.iloc[:, 1:].to_csv(test_data_csv, header=False, index=False)
            
    save_config = SaveConfig(save_interval=frequency)
    hook = Hook(
        out_dir=out_dir,
        train_data=(train_data_csv, "csv"),
        validation_data=(test_data_csv, "csv"),
        save_config=save_config,
        include_collections=collections,
    )

    return hook

def train(session, args):
    train_dir = os.environ.get('SM_CHANNEL_TRAIN', "../data")
    test_dir = os.environ.get('SM_CHANNEL_TEST', "../data")
    print(f"train_dir: {train_dir}")
    print(f"val_dir: {test_dir}")

    train_csv = glob.glob( f"{train_dir}/*.csv")
    test_csv = glob.glob( f"{test_dir}/*.csv")
    
    train_df = pd.read_csv(f"{train_csv[0]}", names=cols)
    test_df = pd.read_csv(f"{test_csv[0]}", names=cols)
    
    run_name = None
    if args.sm_run is not None:
        run_name=f"{args.sm_run}-{strftime('%Y%m%d%H%M%S', gmtime())}"
        
    with load_run(sagemaker_session=session, experiment_name=args.sm_experiment, run_name=run_name) as run:
        x = train_df.drop(["msno"],axis=1).iloc[:, 1:]
        y = train_df.loc[:, "is_churn"]
        
        x_test = test_df.drop(["msno"],axis=1).iloc[:, 1:]
        y_test = test_df.loc[:, "is_churn"]
        
        # The output_uri is a the URI for the s3 bucket where the metrics will be
        # saved.
        output_uri = args.smdebug_path if args.smdebug_path is not None else args.output_uri

        collections = (
            args.smdebug_collections.split(",") if args.smdebug_collections is not None else None
        )

        hook = create_smdebug_hook(
            out_dir=output_uri,
            frequency=args.smdebug_frequency,
            collections=collections,
            train_data=train_df,
            validation_data=test_df,
        )
        
        model = XGBClassifier(objective="binary:logistic", 
                             max_depth=int(args.max_depth), 
                             eta=float(args.eta),
                             gamma=int(args.gamma),
                             min_child_weight=int(args.min_child_weight),
                             subsample=float(args.subsample),
                             n_estimators=int(args.n_estimators),
                             callbacks=[hook])
        
        model.fit(x,y, eval_set=[(x, y), (x_test, y_test)])
        eval_results = model.evals_result()
        train_loss_dict = eval_results['validation_0']
        train_loss_arr = train_loss_dict['logloss']
        for idx, loss in enumerate(train_loss_arr):
            print(f'[{idx:03}]#011train-logloss:{loss};')
            # print(f"train idx: {idx}, loss: {loss}, type: {type(loss)}")
            run.log_metric(name="train:logloss", value=loss, step=idx)
        # print(f"[000]#011train-logloss:{train_loss_arr[-1]};")
    
        val_loss_dict = eval_results['validation_1']
        val_loss_arr = val_loss_dict['logloss']
        for idx, loss in enumerate(val_loss_arr):
            # print(f"val idx: {idx}, loss: {loss}")
            print(f'[{idx:03}]#011validation-logloss:{loss};')
            run.log_metric(name="validation:logloss", value=loss, step=idx)
        # print(f"[000]#011validation-logloss:{val_loss_arr[-1]};")
        print(f"evaluation results: {eval_results}")
        y_pred = model.predict(x_test)
        y_pred_proba=model.predict_proba(x_test)[:,1]

        run.log_confusion_matrix(y_test, y_pred, title="ConfusionMatrix")
        run.log_roc_curve(y_test, y_pred_proba, title="ROCCurve")
        run.log_precision_recall(y_test, y_pred_proba, positive_label=1, title="PrecisionRecall")

    return model
    
def evaluation(session, model):
    val_dir = os.environ.get('SM_CHANNEL_VALIDATION', "../data")
    print(f"val_dir: {val_dir}")
    X_test = pd.read_csv(os.path.join(val_dir,'X_test.csv'))
    y_test = pd.read_csv(os.path.join(val_dir,'y_test.csv'))
    y_pred = model.predict(X_test)
    y_pred_proba=model.predict_proba(X_test)[:,1]
    with load_run(sagemaker_session=session) as run:
        run.log_confusion_matrix(y_test, y_pred, title="ConfusionMatrix")
        run.log_roc_curve(y_test, y_pred_proba, title="ROCCurve")
        run.log_precision_recall(y_test, y_pred_proba, positive_label=1, title="PrecisionRecall")

    print('\nConfusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('\nScores')
    print('------------------------')
    print('AUC:',np.round(roc_auc_score(y_test,y_pred_proba)*100,2),'%')
    print('Accuracy:',np.round(accuracy_score(y_test,y_pred)*100,2),'%')
    print('Precision:',np.round(precision_score(y_test,y_pred)*100,2),'%')
    print('Recall:',np.round(recall_score(y_test,y_pred)*100,2),'%')
    print('F1 score:',np.round(f1_score(y_test,y_pred)*100,2))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser( prog = 'train.py',
                    description = 'A script that trains a model using XGBoost to predict customer churn for a streaming service.')
    parser.add_argument('--max_depth', default=5, help='max depth hyperparameter')
    parser.add_argument('--eta', default=0.2, help="eta hyperparameter")
    parser.add_argument('--gamma', default=4, help="gamma hyperparameter")
    parser.add_argument('--min_child_weight', default=6, help="min child weight hyperparameter")
    parser.add_argument('--subsample', default=0.7, help="subsample hyperparameter")
    parser.add_argument('--n_estimators', default=50, help="number of rounds hyperparameter")
    parser.add_argument('--region', help="AWS region where the training job is run")
    parser.add_argument('--model_dir', default=os.environ.get('SM_MODEL_DIR'), help="AWS region where the training job is run")
    parser.add_argument('--sm_experiment', help="Sagemaker Experiment name associates with the training job")
    parser.add_argument('--sm_run', help="Sagemaker Experiment Trial name associates with the training job")
    parser.add_argument("--smdebug_path", type=str, default=None)
    parser.add_argument("--smdebug_frequency", type=int, default=1)
    parser.add_argument("--smdebug_collections", type=str, default="metrics")
    parser.add_argument("--output_uri", type=str, default="/opt/ml/output/tensors", help="S3 URI of the bucket where tensor data will be stored.")

    args = parser.parse_args()
    
    sagemaker_session = Session(boto_session=boto3.session.Session(region_name=args.region))
    model = train(sagemaker_session, args)
    model_location = args.model_dir + '/xgboost-churn-model.json'
    model.save_model(model_location)
    # pkl.dump(model, open(model_location, 'wb'))
