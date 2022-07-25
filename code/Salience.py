from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import MultiLabelClassificationExplainer
import pandas as pd
from numpy import load
from csv import writer
import os

# Prepare data set
# data should contains test data AND result column.
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cohort2.csv")

compliant = data[data["labels"] == 0]
non = data[data["labels"] == 1]

# Divide the data into 4 sets
falsepos = compliant[compliant["sepResult"] == 1]
trueneg = compliant[compliant["sepResult"] == 0]
falseneg = non[non["sepResult"]==0]
truepos = non[non["sepResult"]==1]

# Select a set of to inspect
data = truepos

# Prepare model and explainer
model_name = "/content/drive/MyDrive/Colab Notebooks/SEP_MODEL"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)

# Select an email to inspect
index = 21

# Prepare text
text = data.iloc[index]["concatText"].replace('[SEP]', ',')
id = data.iloc[index]["Artifact ID"]
reason = data.iloc[index]["First Line Review Reason"]

#print("true: "+str(data.iloc[index]["labels"]))
#print("predict: "+str(data.iloc[index]["sepResult"]))
#print(text)
#print(id)
#print(reason)

# Get salience
cls_explainer.visualize()