import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

NPZ_PATH = '../db/stereo_after_rir/'

class MelDataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            child_filepath = NPZ_PATH + 'train_child_8k_mono.npz'
            male_filepath = NPZ_PATH + 'train_male_8k_mono.npz'
            female_filepath = NPZ_PATH + 'train_female_8k_mono.npz'
        elif mode == 'val':
            child_filepath = NPZ_PATH + 'val_child_8k_mono.npz'
            male_filepath = NPZ_PATH + 'val_male_8k_mono.npz'
            female_filepath = NPZ_PATH + 'val_female_8k_mono.npz'

        with np.load(male_filepath) as data:
            male_x_data = data['x']
            data_len = len(male_x_data)

            male_y_data = np.array(np.zeros([data_len]) + 0, dtype=np.int32).reshape(-1)  # male : 0
            print(male_x_data.shape, male_y_data.shape)

        with np.load(female_filepath) as data:
            female_x_data = data['x']
            data_len = len(female_x_data)

            female_y_data = np.array(np.zeros([data_len]) + 1, dtype=np.int32).reshape(-1)  # female : 1
            print(female_x_data.shape, female_y_data.shape)

        with np.load(child_filepath) as data:
            child_x_data = data['x']
            data_len = len(child_x_data)

            child_y_data = np.array(np.zeros([data_len]) + 2, dtype=np.int32).reshape(-1) # child : 2
            print(child_x_data.shape, child_y_data)

        self.x_data = np.array(np.concatenate((male_x_data, female_x_data, child_x_data), axis=0), dtype=np.float32)
        self.y_data = np.array(np.concatenate((male_y_data, female_y_data, child_y_data), axis=0), dtype=np.int32)
        print(self.x_data.shape, self.y_data.shape)
        print('data loading is done.')

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        input, target = self.x_data[idx], self.y_data[idx]
        return torch.FloatTensor(input).view(1, 128, 96), torch.LongTensor(target)


if __name__ == "__main__":
    train_dataset = MelDataset(mode='train')
    print(train_dataset)

    data_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
    for batch_idx, samples in enumerate(data_loader):
        data, target = samples
        print(data.shape, target.shape)
        print(data.dtype, target.dtype)

        break