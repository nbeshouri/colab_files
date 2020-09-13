from transformers import *


TOKENIZERS = {
    'bert-base-uncased': BertTokenizer,
    'xlnet-base-cased': XLNetTokenizer,
    'distilbert-base-uncased': DistilBertTokenizer,
    'simple_rnn': BertTokenizer
}


def get_tokenizer(config):

    try:
        return AutoTokenizer.from_pretrained(config.tokenizer)
    except OSError:
        pass

    tokenizer = TOKENIZERS[config.tokenizer]
    return tokenizer(config)
