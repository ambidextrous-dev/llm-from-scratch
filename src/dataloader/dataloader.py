import tiktoken

from torch.utils.data import DataLoader
from src.dataset.gpt_dataset import GPTDatasetV1


class GPTDataLoader:
    def __init__(self, txt, batch_size,  max_length, stride, shuffle, drop_last=True, num_workers=0):
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
            drop_last=drop_last,        #drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training
            num_workers=num_workers     #The number of CPU processes to use for preprocessing
        )

        return dataloader

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)