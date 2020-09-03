import torch
from torch.optim.adam import Adam
from torch.optim import RMSprop
import wandb
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
import argparse
import os
import yaml

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns; sns.set()
from time import time
from classification import models, datasets, tokenizers


class Timer:

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start


def run_model_on_dataset(model, dataloader, config, optimizer=None, scheduler=None):
    total_loss = 0
    preds = []
    logits = []
    label_ids = []

    for i, batch in enumerate(dataloader):
        device = torch.device(config.device)
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3]
        }

        # Not all models will take all args.
        inputs = {
            k: v for k, v in inputs.items()
            if k in model.forward.__code__.co_varnames}

        outputs = model(**inputs)
        loss, batch_logits = outputs[:2]  # See BertForSequenceClassification.forward
        total_loss += loss.item() * len(batch[0])  # Convert from mean to sum.

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_logits = batch_logits.detach().cpu().numpy()
        logits.append(batch_logits)
        preds.extend(np.argmax(batch_logits, axis=1))
        label_ids.extend(inputs["labels"].detach().cpu().numpy())

    logits = np.concatenate(logits, axis=0)

    return logits, preds, label_ids, total_loss / len(dataloader.dataset)


def train_on_dataset(model, dataset, config):
    model.train()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, pin_memory=True)

    if config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.lr)
    else:
        raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')

    scheduler = None
    if config.get('learning_rate_decay_schedule', None) is not None:
        if config.learning_rate_decay_schedule == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(dataloader) * config.epochs)
        else:
            raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')

    return run_model_on_dataset(model, dataloader, config, optimizer=optimizer, scheduler=scheduler)


def eval_on_dataset(model, dataset, config):
    model.eval()
    dataloader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size, pin_memory=True)
    with torch.no_grad():
        return run_model_on_dataset(model, dataloader, config)


def train(config):
    model = models.get_model(config)
    data = datasets.get_dataset(config)
    device = torch.device(config.device)
    model.to(device)

    last_train_preds = None

    for epoch in range(1, config.epochs + 1):
        with Timer() as train_timer:
            train_logits, train_preds, train_label_ids, train_loss = train_on_dataset(model, data.train, config)
        with Timer() as val_timer:
            val_logits, val_preds, val_label_ids, val_loss = eval_on_dataset(model, data.val, config)

        log_dict = {
            'train_accuracy': accuracy_score(train_label_ids, train_preds),
            'train_accuracy_weighted': weighted_accuracy_score(train_label_ids, train_preds),
            'train_loss': train_loss,
            'train_examples_per_second': len(data.train) / train_timer.interval,
            'train_auc': roc_auc_score(train_label_ids, train_logits[:, 1]),

            'val_accuracy': accuracy_score(val_label_ids, val_preds),
            'val_accuracy_weighted': weighted_accuracy_score(val_label_ids, val_preds),
            'val_loss': val_loss,
            'val_examples_per_second': len(data.train) / val_timer.interval,
            'val_auc': roc_auc_score(val_label_ids, val_logits[:, 1]),

            'train_preds_match': int(last_train_preds is None or tuple(train_preds) == last_train_preds),
            'train_preds_count': len(train_preds),
            'train_preds_mean': np.average(train_preds)
        }
        wandb.log(log_dict)
        print(log_dict)
        last_train_preds = tuple(train_preds)


def weighted_accuracy_score(y_true, y_pred):
    class_to_weight = pd.Series(y_true).value_counts()
    class_to_weight = class_to_weight.max() / class_to_weight

    sample_weights = [class_to_weight[c] for c in y_true]
    return accuracy_score(y_true, y_pred, sample_weight=sample_weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configs', type=str, default=None, metavar='N')
    parser.add_argument('--project', type=str, default='delete_me', metavar='N')

    args, unknown = parser.parse_known_args()

    if args.configs is not None:
        os.environ['WANDB_CONFIG_PATHS'] = args.configs

    wandb.init(entity='nbeshouri', project=args.project)

    config = wandb.config

    if torch.cuda.is_available():
        config.device = 'cuda'
    else:
        config.device = 'cpu'

    train(config)
