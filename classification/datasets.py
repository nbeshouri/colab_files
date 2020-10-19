import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import pandas as pd
import os
from . import DATA_DIR_PATH, tokenizers


def prepare_tensor_dataset(texts, class_ids, max_seq_length, tokenizer):
    """Convert a sequence of texts and labels to a dataset."""

    class_ids = list(
        class_ids
    )  # HACK: Some how the fact that this one is a series breaks shit.

    def pad_seqs(seqs):
        return [seq + ((max_seq_length - len(seq)) * [0]) for seq in seqs]

    token_ids_seqs = [
        tokenizer.encode(
            t, add_special_tokens=True, max_length=max_seq_length, truncation=True
        )
        for t in texts
    ]

    att_masks = [[1] * len(ts) for ts in token_ids_seqs]
    token_type_id_seqs = [[0] * len(ts) for ts in token_ids_seqs]

    token_ids_seqs = torch.tensor(pad_seqs(token_ids_seqs), dtype=torch.long)
    att_masks = torch.tensor(pad_seqs(att_masks), dtype=torch.long)
    token_type_id_seqs = torch.tensor(pad_seqs(token_type_id_seqs), dtype=torch.long)
    class_ids = torch.tensor(class_ids, dtype=torch.long)

    return TensorDataset(token_ids_seqs, att_masks, token_type_id_seqs, class_ids)


def under_sample(data_df, class_col, random_state=42):
    value_counts = data_df[class_col].value_counts()
    count = value_counts.min()
    df_parts = []
    for class_i in value_counts.index:
        df_parts.append(
            data_df[data_df[class_col] == class_i].sample(
                count, random_state=random_state
            )
        )
    sample_df = pd.concat(df_parts, axis=0)
    return sample_df.sample(sample_df.shape[0], random_state=42)


# def load_dataset(dataset_name, version, tokenizer, max_length):
#     if dataset_name == 'wikipedia_comments':
#         train_df = pd.read_csv(os.path.join(DATA_DIR_PATH, 'datasets', dataset_name, 'train.csv.zip'))
#         test_df = pd.read_csv(os.path.join(DATA_DIR_PATH, 'datasets', dataset_name, 'test.csv.zip'))
#         test_labels_df = pd.read_csv(os.path.join(DATA_DIR_PATH, 'datasets', dataset_name, 'test_labels.csv.zip'))
#         test_df = pd.merge(test_df, test_labels_df)
#         if version == 'train':
#             texts = train_df[:-5000].comment_text
#             target_classes = train_df[:-5000].toxic
#         elif version == 'val':
#             texts = train_df[-5000:].comment_text
#             target_classes = train_df[-5000:].toxic
#         elif version == 'test':
#             texts = test_df.comment_text
#             target_classes = abs(test_df.toxic)  # They're -1 for some reason.
#     elif dataset_name == 'newsgroups':
#         from sklearn.datasets import fetch_20newsgroups
#         cats = ['alt.atheism', 'sci.space']
#         if version == 'train':
#             newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
#             texts = newsgroups_train.data
#             target_classes = newsgroups_train.target
#         elif version == 'val':
#             newsgroups_train = fetch_20newsgroups(subset='test', categories=cats)
#             texts = newsgroups_train.data
#             target_classes = newsgroups_train.target
#     return get_dataset(texts, target_classes, max_length, tokenizer)


class Datasets:
    name = None


class WikiCommentsDatasets(Datasets):

    name = "wikipedia_comments"

    def __init__(self, tokenizer, config):
        train_df = pd.read_csv(
            os.path.join(
                DATA_DIR_PATH, "datasets", "wikipedia_comments", "train.csv.zip"
            )
        )
        test_df = pd.read_csv(
            os.path.join(
                DATA_DIR_PATH, "datasets", "wikipedia_comments", "test.csv.zip"
            )
        )
        test_labels_df = pd.read_csv(
            os.path.join(
                DATA_DIR_PATH, "datasets", "wikipedia_comments", "test_labels.csv.zip"
            )
        )
        test_df = pd.merge(test_df, test_labels_df)
        test_df = test_df.query("toxic != -1")

        if config.get("undersample", False):
            train_df = under_sample(train_df, class_col="toxic")
            test_df = under_sample(test_df, class_col="toxic")

        self.train = prepare_tensor_dataset(
            texts=train_df.comment_text[: config.max_train_size],
            class_ids=train_df.toxic[: config.max_train_size],
            max_seq_length=config.max_seq_length,
            tokenizer=tokenizer,
        )
        self.val = prepare_tensor_dataset(
            texts=test_df.comment_text[: config.max_val_size],
            class_ids=test_df.toxic[: config.max_val_size],
            max_seq_length=config.max_seq_length,
            tokenizer=tokenizer,
        )


class NewsgroupDatasets(Datasets):

    name = "newsgroups"

    def __init__(self, tokenizer, config):
        from sklearn.datasets import fetch_20newsgroups

        cats = ["alt.atheism", "sci.space"]
        newsgroups_train = fetch_20newsgroups(subset="train", categories=cats)
        newsgroups_test = fetch_20newsgroups(subset="test", categories=cats)
        self.train = prepare_tensor_dataset(
            texts=newsgroups_train.data[: config.max_train_size],
            class_ids=newsgroups_train.target[: config.max_train_size],
            max_seq_length=config.max_seq_length,
            tokenizer=tokenizer,
        )
        self.val = prepare_tensor_dataset(
            texts=newsgroups_test.data[: config.max_val_size],
            class_ids=newsgroups_test.target[: config.max_val_size],
            max_seq_length=config.max_seq_length,
            tokenizer=tokenizer,
        )


def get_dataset(config):
    tokenizer = tokenizers.get_tokenizer(config)
    for obj in globals().values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Datasets)
            and obj.name == config.dataset
        ):
            return obj(tokenizer, config)
