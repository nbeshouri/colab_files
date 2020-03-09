from transformers import BertTokenizer


def get_tokenizer(config):
    if 'bert' in config.tokenizer:
        return BertTokenizer.from_pretrained(
            config.tokenizer,
            do_lower_case='uncased' in config.tokenizer)
