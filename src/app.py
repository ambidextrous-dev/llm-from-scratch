import tiktoken

from src.service.BuildVocabulary import BuildVocabulary
from src.tokenizer.SimpleTokenizerV1 import SimpleTokenizerV1
from src.tokenizer.SimpleTokenizerV2 import SimpleTokenizerV2

vBuilder = BuildVocabulary()
vBuilder.build_vocab()

print("-----------------------------")
print("Testing SimpleTokenizer V1")
tokenizer = SimpleTokenizerV1(vBuilder.vocab)
text = """"It's the last he painted, you know,"
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

# Fails for SimpleTokenizerV1, as Hello is not present in the vocabulary
print("-----------------------------")
print("Testing SimpleTokenizer V2")
tokenizerV2 = SimpleTokenizerV2(vBuilder.vocab)
text = "Hello, do you like tea?"
v2_ids = tokenizerV2.encode(text)
print(v2_ids)
print(tokenizerV2.decode(v2_ids))

# BPE based tokenizer
print("-----------------------------")
print("Testing BPE Tokenizer")
bpe_tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace like Akwirw"
)
integers = bpe_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = bpe_tokenizer.decode(integers)
print(strings)