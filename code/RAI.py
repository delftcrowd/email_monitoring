import pandas as pd
import numpy as np
from numpy import load
from raiwidgets import ErrorAnalysisDashboard

# Define test data path and result path
test_path = "/content/drive/MyDrive/Colab Notebooks/test.csv"
res_path = "/content/drive/MyDrive/Colab Notebooks/SEP_result.npy"


test_data = pd.read_csv(test_path)
true_y = test_data['labels']
test_data = test_data.drop(['text', 'Artifact ID','First Line Review Reason', 'concatText','Unnamed: 0', 'labels'], axis=1)
print(test_data.columns)# test_data.columns and feature_names should be the same, order matters.
feature_names = ['#Receiver', 'receiverPers', 'senderPers','posInChain', 'length']
predictions = load(res_path)


ErrorAnalysisDashboard(dataset=test_data, true_y=true_y, features=feature_names, pred_y=predictions)