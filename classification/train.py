import torch
from torch.optim.adam import Adam
from torch.optim import RMSprop
import wandb
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns; sns.set()
from time import time
from . import models, datasets, tokenizers


class Timer:

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start


def run_model_on_dataset(model, dataloader, config):
    total_loss = 0
    preds = []
    label_ids = []

    if model.training:
        if config.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == 'rmsprop':
            optimizer = RMSprop(model.parameters(), lr=config.lr)

    for i, batch in enumerate(dataloader):
        device = torch.device(config.device)
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }
        outputs = model(**inputs)
        loss, logits = outputs[:2]  # See BertForSequenceClassification.forward
        total_loss += loss.item() * len(batch[0])  # Convert from mean to sum.

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=1))
        label_ids.extend(inputs["labels"].detach().cpu().numpy())

    return preds, label_ids, total_loss / len(dataloader.dataset)


def train_on_dataset(model, dataset, config):
    model.train()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, pin_memory=True)
    return run_model_on_dataset(model, dataloader, config)


def eval_on_dataset(model, dataset, config):
    model.eval()
    dataloader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size, pin_memory=True)
    with torch.no_grad():
        return run_model_on_dataset(model, dataloader, config)


def train(config, wb_project=None):
    model = models.get_model(config)
    data = datasets.get_dataset(config)
    device = torch.device(config.device)
    model.to(device)

    if wb_project is not None:
        wandb.init(entity='nbeshouri', project=wb_project)
        wandb.watch_called = False
        wandb.watch(model, log=None, log_freq=1)
        wandb.config.update(config)

    last_train_preds = None

    for epoch in range(config.epochs + 1):
        with Timer() as train_timer:
            if epoch == 0:
                train_preds, train_label_ids, train_loss = eval_on_dataset(model, data.train, config)
            else:
                train_preds, train_label_ids, train_loss = train_on_dataset(model, data.train, config)
        with Timer() as val_timer:
            val_preds, val_label_ids, val_loss = eval_on_dataset(model, data.val, config)

        if wb_project is not None:
            log_dict = {
                'train_accuracy': accuracy_score(train_label_ids, train_preds),
                'train_loss': train_loss,
                'train_examples_per_second': len(data.train) / train_timer.interval,
                'val_accuracy': accuracy_score(val_label_ids, val_preds),
                'val_loss': val_loss,
                'val_examples_per_second': len(data.train) / val_timer.interval,
                'train_preds_match': int(last_train_preds is None or tuple(train_preds) == last_train_preds),
                'train_preds_count': len(train_preds),
                'train_preds_mean': np.average(train_preds)
            }
            wandb.log(log_dict)
            print(log_dict)
            last_train_preds = tuple(train_preds)
