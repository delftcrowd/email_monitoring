import pandas as pd
import numpy as np
from numpy import save
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer





trainSet = "/content/drive/MyDrive/Colab Notebooks/PUREtrain.csv"
testSet = "/content/drive/MyDrive/Colab Notebooks/PUREtest.csv"
modelSaveTo = "/content/drive/MyDrive/Colab Notebooks/PURE_MODEL"
resultSaveTo = "/content/drive/MyDrive/Colab Notebooks/PURE_result"



training_args = TrainingArguments(
    output_dir= modelSaveTo,
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    num_train_epochs= 3,
    weight_decay=0.01,
)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation = True, padding='max_length') # truncation = True disabled.



def trainModelSave():

    dataset = load_dataset('csv', data_files={'train': trainSet,
                                              'test': testSet})
 
    dataset = dataset.remove_columns(['Unnamed: 0'])
    print(dataset)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    trainer = Trainer(
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2),
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print('now train')
    trainer.train()
    print(trainer.save_model())


def modelPredict():

    dataset = load_dataset('csv', data_files={'train': trainSet,
                                              'test': testSet})
    dataset = dataset.remove_columns(['Unnamed: 0'])
    print(dataset)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    trainer = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(modelSaveTo),
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print('now predict')
    metric = trainer.predict(tokenized_dataset["test"])

    result = metric.predictions
    prob = softmax(result, axis=1)
    predictedLabels = np.argmax(prob, axis=1)

    save('/content/drive/MyDrive/Colab Notebooks/PURE_result', predictedLabels)
    print(metric)


if __name__ == '__main__':

    # Choose either to train a model or to predict.
    trainModelSave()
    #modelPredict()
