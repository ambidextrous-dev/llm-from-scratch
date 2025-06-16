import re


# Tokenizes and Detokenizes a text based on the already trained vocabulary
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        # split words on punctuations
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        # remove whitespaces
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# Tokenizes and Detokenizes a text based on the already trained vocabulary
# V2 - also able to handle unknown words
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        # split words on punctuations
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        # remove whitespaces
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]

        # handle unknown word
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
