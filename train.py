import argparse
import os
from time import perf_counter, sleep, time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
import yaml
from classification import datasets, models, tokenizers
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import RMSprop
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import get_linear_schedule_with_warmup

sns.set()


TEMP_WEIGHTS_PATH = "state_dict.pickle"


class Timer:
    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start


def run_model_on_dataset(
    model, dataloader, config, yield_freq=None, optimizer=None, scheduler=None
):
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
            "labels": batch[3],
        }

        # Not all models will take all args.
        inputs = {
            k: v for k, v in inputs.items() if k in model.forward.__code__.co_varnames
        }

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

        if (
            i == len(dataloader) - 1
            or yield_freq is not None
            and (i + 1) % yield_freq == 0
        ):
            logits = np.concatenate(logits, axis=0)
            yield logits, preds, label_ids, total_loss / batches_since_yield
            total_loss = 0
            preds = []
            logits = []
            label_ids = []
            batches_since_yield = 0


def train(config, run):
    model = models.get_model(config)
    if config.get("log") is not None:
        wandb.watch(model, log=config.get("log"))
    data = datasets.get_dataset(config)
    device = torch.device(config.device)
    model.to(device)

    best_performance = None
    best_performance_metrics = None
    step = 0
    for epoch in range(1, config.epochs + 1):
        if config.optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == "rmsprop":
            optimizer = RMSprop(model.parameters(), lr=config.lr)
        else:
            raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')

        dataloader = DataLoader(
            data.train, shuffle=True, batch_size=config.batch_size, pin_memory=True
        )

        scheduler = None
        if config.get("learning_rate_decay_schedule", None) is not None:
            if config.learning_rate_decay_schedule == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=len(dataloader) * config.epochs,
                )
            else:
                raise ValueError(f'"{config.optimizer}" is an invalid optimizer name!')
        model.train()
        mini_batch_start_time = perf_counter()

        for logits, preds, label_ids, loss in run_model_on_dataset(
            model,
            dataloader,
            config,
            yield_freq=config.get("log_freq"),
            optimizer=optimizer,
            scheduler=scheduler,
        ):
            step += 1
            train_metrics = compute_metrics(
                logits=logits,
                preds=preds,
                label_ids=label_ids,
                loss=loss,
                runtime=perf_counter() - mini_batch_start_time,
            )
            log_run("train", train_metrics, step=step, epoch=epoch)

            # Validate
            model.eval()
            dataloader = DataLoader(
                data.val, shuffle=False, batch_size=config.batch_size, pin_memory=True
            )
            with torch.no_grad():
                start_time = perf_counter()
                logits, preds, label_ids, loss = iter(
                    next(
                        run_model_on_dataset(model, dataloader, config, yield_freq=None)
                    )
                )
                val_metrics = compute_metrics(
                    logits=logits,
                    preds=preds,
                    label_ids=label_ids,
                    loss=loss,
                    runtime=perf_counter() - start_time,
                )
                log_run("val", val_metrics, step=step, epoch=epoch)

                if config.checkpoint_metric is not None:
                    if (
                        best_performance is None
                        or val_metrics[config.checkpoint_metric] > best_performance
                    ):
                        best_performance = val_metrics[config.checkpoint_metric]
                        best_performance_metrics = val_metrics
                        torch.save(model.state_dict(), TEMP_WEIGHTS_PATH)

            model.train()  # Need to re-enter training model.

            mini_batch_start_time = perf_counter()

    if config.checkpoint_metric is not None:
        # Relog the best validation metrics because W&B
        # seems to want to sort by that in sweeps.
        log_run("val", best_performance_metrics)

        # Save the best model weights.
        artifact = wandb.Artifact(
            f"{run.name.replace('-', '_')}_best_weights", type="weights"
        )
        artifact.add_file(TEMP_WEIGHTS_PATH)
        run.log_artifact(artifact)


def weighted_accuracy_score(y_true, y_pred):
    class_to_weight = pd.Series(y_true).value_counts()
    class_to_weight = class_to_weight.max() / class_to_weight

    sample_weights = [class_to_weight[c] for c in y_true]
    return accuracy_score(y_true, y_pred, sample_weight=sample_weights)


def log_run(
    run_type,
    metrics,
    epoch=None,
    **kwargs,
):
    log_dict = {f"{run_type}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        log_dict["epoch"] = epoch
    print(log_dict)
    wandb.log(log_dict, **kwargs)


def compute_metrics(
    logits,
    preds,
    label_ids,
    loss,
    runtime,
):
    return {
        "accuracy": accuracy_score(label_ids, preds),
        "accuracy_weighted": weighted_accuracy_score(label_ids, preds),
        "loss": loss,
        "examples_per_second": len(preds) / runtime,
        "auc": roc_auc_score(label_ids, logits[:, 1]),
        "sample_size": len(preds),
    }


class ConfigWrapper:
    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getattr__(self, key):
        return self.config.get(key, None)

    def __setattr__(self, key, value):
        if key == "config":
            self.__dict__["config"] = value
        else:
            setattr(self.config, key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, default=None, metavar="N")
    parser.add_argument("--project", type=str, default="delete_me", metavar="N")

    args, unknown = parser.parse_known_args()

    if args.configs is not None:
        os.environ["WANDB_CONFIG_PATHS"] = args.configs

    run = wandb.init(entity="nbeshouri", project=args.project)

    config = ConfigWrapper(wandb.config)

    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"

    train(config, run)
