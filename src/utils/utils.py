import urllib.request

def fetch_file():
    # fetch the book
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")

    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

def get_raw_text():
    fetch_file()

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text

def get_target_pairs_sample(tokenizer, encoded_text):
    context_size = 4  # 1

    for i in range(1, context_size + 1):
        context = encoded_text[:i]
        desired = encoded_text[i]
        # print(context, "---->", desired)
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))