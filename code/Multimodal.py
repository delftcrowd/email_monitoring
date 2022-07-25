from dataclasses import dataclass, field
import json
import logging
import os
import numpy as np
from numpy import save
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
from transformers import Trainer, TrainingArguments
from multimodal_transformers.data import load_data
from multimodal_transformers.model import AutoModelWithTabular, TabularConfig
from transformers import AutoConfig
from scipy.special import softmax

if __name__ == '__main__':

  # Define the paths of train/test/val dataset and create dataframes accordingly
  
    path_test = '/content/drive/MyDrive/Colab Notebooks/test.csv'
    path_train = '/content/drive/MyDrive/Colab Notebooks/train.csv'
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

  # Make column info
    text_cols = ['text']
    label_col = 'labels'
    categorical_cols = ['senderPers', 'receiverPers']
    numerical_cols = ['#Reveiver', 'length', 'posInChain']
    label_list = ['Compliant', 'NonCompliant']  # what each label class represents

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  # make sure NaN values for cat columns are filled before passing to load_data
    for c in categorical_cols:
        train.loc[:, c] = train.loc[:, c].astype(str).fillna("-9999999")
        test.loc[:, c] = test.loc[:, c].astype(str).fillna("-9999999")

    column_info_dict = {
        'text_cols': text_cols,
        'num_cols': numerical_cols,
        'cat_cols': categorical_cols,
        'label_col': label_col,
        'label_list': label_list
    }


  # Create torch_datasets - input format required by multimodal transformers
    torch_dataset_train = load_data(
        train,
        text_cols,
        tokenizer,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        label_col = label_col,
        sep_text_token_str=tokenizer.sep_token
    )

    torch_dataset_test = load_data(
        test,
        text_cols,
        tokenizer,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        label_col = label_col,
        sep_text_token_str=tokenizer.sep_token
    )

    #num_labels = len(np.unique(torch_dataset, labels))

  # add tabular_cofig to autoconfig, pass to model
    config = AutoConfig.from_pretrained('bert-base-uncased')

    tabular_config = TabularConfig(
        num_labels = 2,
        cat_feat_dim=torch_dataset_train.cat_feats.shape[1],
        numerical_feat_dim=torch_dataset_train.numerical_feats.shape[1],
        combine_feat_method='weighted_feature_sum_on_transformer_cat_and_numerical_feats',
    )
    config.tabular_config = tabular_config

  # Indicate the path to save the trained model here
    training_args = TrainingArguments(
        output_dir='/content/drive/MyDrive/Colab Notebooks/multimodal_MODEL',
        logging_dir='/content/drive/MyDrive/Colab Notebooks/multi_run',
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=10,
        num_train_epochs=3,
        logging_steps=25,
    )

    trainer = Trainer(

        # For train
        # model = AutoModelWithTabular.from_pretrained('bert-base-uncased', config=config),

        # For test
        model=AutoModelWithTabular.from_pretrained('/content/drive/MyDrive/Colab Notebooks/multimodal_MODEL', config=config),
        args=training_args,
        train_dataset = torch_dataset_train,
    )

  ############# Train a model ##################

    #trainer.train()
    #trainer.save_model()
    

  ############  Run test  ####################

    metric = trainer.predict(torch_dataset_test)
    # result for each data point is the score of each class
    result = metric.predictions 
    prob = softmax(result, axis=1)
    predictedLabels = np.argmax(prob, axis=1)

    save('/content/drive/MyDrive/Colab Notebooks/multi_result', predictedLabels)   # Save the prediction result ([0,1,0 ... 0,0,0]) to path

    print(metric)
