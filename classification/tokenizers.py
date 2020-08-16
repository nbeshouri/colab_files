from transformers import BertTokenizer, DistilBertTokenizer


def get_tokenizer(config):
    if config.tokenizer == 'bert-base-uncased' in config.tokenizer:
        return BertTokenizer.from_pretrained(config.tokenizer)
    elif config.tokenizer == 'distilbert-base-uncased' in config.tokenizer:
        return DistilBertTokenizer.from_pretrained(config.tokenizer)
    else:
        raise ValueError()
