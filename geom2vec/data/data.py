import torch
import numpy as np
import os
from tqdm import tqdm


class Preprocessing:
    """ Preprocess the original trajectories to create datasets for training.

    Parameters
    ----------
    dtype : dtype, default = float
    """

    def __init__(self,
                 torch_or_numpy='numpy',
                 dtype=float):

        self._torch_or_numpy = torch_or_numpy

        if torch_or_numpy == 'torch':
            self._dtype = torch.float32
        else:
            self._dtype = np.float32

        self._dtype = dtype

    def _seq_trajs(self, data):

        if self._torch_or_numpy == 'numpy':
            data = data.copy()
            if not isinstance(data, list):
                data = [data]
            for i in range(len(data)):
                data[i] = data[i].astype(self._dtype)

        else:
            if not isinstance(data, list):
                data = [data]
            for i in range(len(data)):
                if not isinstance(data[i], torch.Tensor):
                    data[i] = torch.tensor(data[i], dtype=self._dtype)
                else:
                    data[i] = data[i].clone().detach().to(self._dtype)

        return data

    def create_dataset(self, data, lag_time):
        """ Create the dataset as the input to VAMPnets.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        lag_time : int
            The lag_time used to create the dataset consisting of time-instant and time-lagged data.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has two elements: one is the instantaneous data frame, the other is the corresponding time-lagged data frame.
        """

        data = self._seq_trajs(data)

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time
            for i in range(L_re):
                dataset.append((data[k][i, :], data[k][i + lag_time, :]))

        return dataset

    def load_dataset(self,data_path, mmap_mode='r'):
        """ Load the dataset from the file.

        Parameters
        ----------
        data_path : str
            The path to the file.
            The type of data to be stored in the output file.
        mmap_mode: str, default = 'r'
            The mode to open the file. If None, the file will be opened in the default mode.

        Returns
        -------
        data : list
            The dataset.
        """

        # iterate over the .npz files and load the data

        assert mmap_mode in ['r', 'r+', 'w+', 'c', None]

        if self._torch_or_numpy == 'torch':
            files = os.listdir(data_path)
            files = [os.path.join(data_path,f) for f in files if f.endswith('.pt')]
            files = sorted(files,key=lambda x: int(x.split('_')[-1].split('.')[0]))

        else:
            files = os.listdir(data_path)
            files = [os.path.join(data_path,f) for f in files if f.endswith('.npz')]
            files = sorted(files,key=lambda x: int(x.split('_')[-1].split('.')[0]))

        data = []
        for file in tqdm(files):
            if self._torch_or_numpy == 'torch':
                data.append(torch.load(file, map_location='cpu', mmap=mmap_mode))
            else:
                data.append(np.load(file, mmap_mode=mmap_mode)['arr_0'])

        return data
