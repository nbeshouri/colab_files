import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from transformers import *


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


MODELS = {
    'bert-base-uncased': BertModel,
    'gpt2': GPT2Model,
    'xlnet-base-cased': XLNetModel,
    'distilbert-base-uncased': DistilBertModel,
    'simple_rnn': SimpleModel
}


def get_model(config):
    if config.model_name not in MODELS:
        raise ValueError()
    model_class = MODELS['tokenizer']

    if hasattr(model_class, 'from_pretrained'):
        model = model_class.from_pretrained(config.model_name)
    else:
        model = model_class(config)

    return model
