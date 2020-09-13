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
from time import time, perf_counter
from classification import models, datasets, tokenizers


class Timer:

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start


def run_model_on_dataset(model, dataloader, config, yield_freq=None, optimizer=None, scheduler=None):
    total_loss = 0
    preds = []
    logits = []
    label_ids = []
    batches_since_yield = 0

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
        batches_since_yield += 1

        if i == len(dataloader) - 1 or yield_freq is not None and (i + 1) % yield_freq == 0:
            logits = np.concatenate(logits, axis=0)
            yield logits, preds, label_ids, total_loss / batches_since_yield
            total_loss = 0
            preds = []
            logits = []
            label_ids = []
            batches_since_yield = 0

    # # There could be nothing left to yield here if the yield_freq
    # # is a whole multiple the length of the dataset
    # if logits:
    #     logits = np.concatenate(logits, axis=0)
    #     yield logits, preds, label_ids, total_loss / batches_since_yield


def train_on_dataset(model, train_dataset, val_datasets, config):
    model.train()
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, pin_memory=True)

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

    mini_batch_start_time = perf_counter()
    for logits, preds, label_ids, loss in run_model_on_dataset(
            model, dataloader, config, yield_freq=config.get('log_freq'), optimizer=optimizer, scheduler=scheduler):
        log_run(
            run_type='train',
            logits=logits,
            preds=preds,
            label_ids=label_ids,
            loss=loss,
            runtime=perf_counter() - mini_batch_start_time)
        for val_dataset_name, val_dataset in val_datasets.items():
            validate_on_dataset(model, val_dataset, val_dataset_name, config)
        mini_batch_start_time = perf_counter()


def validate_on_dataset(model, dataset, dataset_name, config):
    model.eval()
    dataloader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size, pin_memory=True)
    with torch.no_grad():
        start_time = perf_counter()
        logits, preds, label_ids, loss = iter(next(run_model_on_dataset(model, dataloader, config, yield_freq=None)))
        log_run(
            run_type=dataset_name,
            logits=logits,
            preds=preds,
            label_ids=label_ids,
            loss=loss,
            runtime=perf_counter() - start_time)


def train(config):
    model = models.get_model(config)
    if config.get('log') is not None:
        wandb.watch(model, log=config.get('log'))
    data = datasets.get_dataset(config)
    device = torch.device(config.device)
    model.to(device)

    for epoch in range(1, config.epochs + 1):
        train_on_dataset(model, data.train, {'val': data.val}, config)


def weighted_accuracy_score(y_true, y_pred):
    class_to_weight = pd.Series(y_true).value_counts()
    class_to_weight = class_to_weight.max() / class_to_weight

    sample_weights = [class_to_weight[c] for c in y_true]
    return accuracy_score(y_true, y_pred, sample_weight=sample_weights)


def log_run(run_type, logits, preds, label_ids, loss, runtime):
    log_dict = {
        'accuracy': accuracy_score(label_ids, preds),
        'accuracy_weighted': weighted_accuracy_score(label_ids, preds),
        'loss': loss,
        'examples_per_second': len(preds) / runtime,
        'auc': roc_auc_score(label_ids, logits[:, 1]),
        'sample_size': len(preds)
    }
    log_dict = {f'{run_type}_{k}': v for k, v in log_dict.items()}
    print(log_dict)
    wandb.log(log_dict)


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
