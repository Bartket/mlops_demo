import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from dagster import op, job, Out
import mlflow
from dagster_mlflow import end_mlflow_on_run_finished, mlflow_tracking

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


mlflow.set_tracking_uri("http://localhost:5000")


@op()
def load_data() -> pd.DataFrame:
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s",
            e,
        )
    return data


@op(
    out={
        "train_x": Out(pd.DataFrame),
        "test_x": Out(pd.DataFrame),
        "train_y": Out(pd.DataFrame),
        "test_y": Out(pd.DataFrame),
    }
)
def train_test_split_data(data: pd.DataFrame):
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    return train_x, test_x, train_y, test_y


@op(
    required_resource_keys={"mlflow"}, config_schema={"alpha": float, "l1_ratio": float}
)
def train_model(
    context,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_y: pd.DataFrame,
):
    alpha = context.op_config["alpha"]
    l1_ratio = context.op_config["l1_ratio"]
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    signature = infer_signature(test_x, lr.predict(test_x))
    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="sklearn-model",
        registered_model_name="ElasticNet-wine",
        signature=signature,
    )


@end_mlflow_on_run_finished
@job(resource_defs={"mlflow": mlflow_tracking})
def mlf_example():
    data = load_data()
    train_x, test_x, train_y, test_y = train_test_split_data(data)
    train_model(train_x, test_x, train_y, test_y)
