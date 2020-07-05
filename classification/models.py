import torch
from torch.optim.adam import Adam
from torch.optim import RMSprop
import wandb
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
from time import time


class SimpleModel(nn.Module):

    name = 'simple_rnn'

    def __init__(self, config):
        super().__init__()
        self.embbeding = nn.Embedding(30552, 768)
        self.rnn = nn.LSTM(768, 768, batch_first=True)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        x = self.embbeding(input_ids)
        x, _ = self.rnn(x)
        logits = self.classifier(x[:, -1])

        output = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            output = (loss,) + output

        return output


def get_model(config):

    if 'bert' in config.model_name:
        return BertForSequenceClassification.from_pretrained(config.model_name)

    for obj in globals().values():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and hasattr(obj, 'name') and obj.name == config.model_name:
            return obj(config)
