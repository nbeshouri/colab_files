from transformers import *


TOKENIZERS = {
    'bert-base-uncased': BertTokenizer,
    'xlnet-base-cased': XLNetTokenizer,
    'distilbert-base-uncased': DistilBertTokenizer,
    'simple_rnn': BertTokenizer
}


def get_tokenizer(config):
    if config.tokenizer not in TOKENIZERS:
        raise ValueError()
    tokenizer_class = TOKENIZERS[config.tokenizer]

    if hasattr(tokenizer_class, 'from_pretrained'):
        tokenizer = tokenizer_class.from_pretrained(config.tokenizer)
    else:
        tokenizer = tokenizer_class(config)

    return tokenizer
