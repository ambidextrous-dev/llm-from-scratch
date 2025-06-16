import re

from src.utils.utils import fetch_file

class BuildVocabulary:
    def __init__(self):
        self.vocab = None

    def build_vocab(self):
        fetch_file()

        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()

        # generate tokens
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # convert unique tokens to token ID's
        all_tokens = sorted(set(preprocessed))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"]) # add two special tokens for unknow and end of text

        # build vocab
        self.vocab = {token:index for index, token in enumerate(all_tokens)}