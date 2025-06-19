import tiktoken

from torch.utils.data import DataLoader
from src.dataset.gpt_dataset import GPTDatasetV1


class GPTDataLoader:
    """
        A wrapper class that creates a PyTorch DataLoader for GPT-style language modeling tasks.
        It uses tiktoken for tokenization and a custom GPTDatasetV1 to generate input-target token sequences
        based on a sliding window over the tokenized text.
    """
    def __init__(self, txt, batch_size,  max_length, stride, shuffle, drop_last=True, num_workers=0):
        """
           Initializes the GPTDataLoader with given configuration parameters.

           Args:
               txt (str): Raw text input to be tokenized and used for training.
               batch_size (int): Number of input-target pairs per batch.
               max_length (int): Length (in tokens) of each input sequence.
               stride (int): Step size to move the sliding window across the tokenized text. Smaller strides result in more overlapping sequences.
               shuffle (bool): Whether to shuffle the dataset during training.
               drop_last (bool): If True, drops the last batch if it is smaller than batch_size. Helps avoid inconsistent loss during training.
               num_workers (int): Number of subprocesses to use for data loading (set to 0 for debugging).
        """
        self.dataloader = self._create_dataloader_v1(txt, batch_size, max_length, stride, shuffle, drop_last, num_workers)

    def _create_dataloader_v1(self, txt, batch_size=4, max_length=256,
                             stride=128, shuffle=True, drop_last=True,
                             num_workers=0):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)