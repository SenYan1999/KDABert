import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

class PretrainedDataset(Dataset):
    def __init__(self, h2file: str, num: int, max_pred_len: int):
        self.input = self.random_choice(h2file, num)
        self.max_pred_len = max_pred_len

    def random_choice(self, h2file, num):
        file = h5py.File(h2file, 'r')
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', \
                'next_sentence_labels']

        input = [np.asarray(file[key][:num]) for key in keys]

        file.close()

        return input

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_position, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.input)
        ]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_len
        padded_mask_indices = (masked_lm_position == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_position[:index]] = masked_lm_ids[:index]

        return [input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels]

    def __len__(self):
        return len(self.input[0])
