import tiktoken
from src.utils.utils import get_raw_text, get_target_pairs_sample

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(get_raw_text())

# temporarily remove first 50 chars - just for demo
enc_sample = enc_text[50:]

# manually generate target pairs
get_target_pairs_sample(tokenizer, enc_sample)