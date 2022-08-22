import mlflow

# demo to run the model via mlflow model repository

logged_model = "runs:/d36c70a33a6c43aaa215b9b01acf18c7/sklearn-model"  # replace this with saved model

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = pd.read_csv("demo/data/winequality-red.csv")
prediction = data.iloc[:1, :-1]
label = data.iloc[:1, -1]

print("Data: \n " + str(prediction))
print("Prediction: \n " + str(loaded_model.predict(pd.DataFrame(prediction))))
print("Label: \n " + str(label.values))
