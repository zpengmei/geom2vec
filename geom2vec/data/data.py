import os

import numpy as np
import torch
from tqdm import tqdm
from copy import copy
from torch.utils.data import Dataset


class Preprocessing:
    """
    Preprocess the original trajectories to create datasets for training.

    Parameters
    ----------
    torch_or_numpy
        Array type of the elements of the output dataset.

    """

    def __init__(
            self,
            torch_or_numpy="numpy",
    ):
        self._torch_or_numpy = torch_or_numpy

        if torch_or_numpy == "torch":
            self._dtype = torch.float32
        else:
            self._dtype = np.float32

    def _seq_trajs(self, data):
        if self._torch_or_numpy == "numpy":
            data = copy(data)
            if not isinstance(data, list):
                data = [data]
            for i in range(len(data)):
                data[i] = data[i].astype(self._dtype)

        else:
            if not isinstance(data, list):
                data = [data]
            for i in range(len(data)):
                if not isinstance(data[i], torch.Tensor):
                    data[i] = torch.tensor(data[i])
                else:
                    data[i] = data[i].clone().detach()

        return data

    def create_time_lagged_dataset(self, data, lag_time):
        """
        Create a time-lagged dataset.

        This dataset is used for VAMPnet/SRV training/validation.

        Parameters
        ----------
        data : list or ndarray or torch.Tensor
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

    def create_time_lagged_state_label_dataset(self, data, data_weights, state_labels, lag_time):
        """
        create a time-lagged dataset for SPIB, user provides the input feature and correspoding state labels
        SPIB take instantaneous data and predict the future state labels

        Args:
            data: list or ndarray or torch.Tensor
            state_labels: list or ndarray or torch.Tensor
            data_weights: list or ndarray or torch.Tensor
            lag_time: int
        Returns:
            dataset: list of tuples, each tuple has two elements: one is the instantaneous data frame,
            the other is the corresponding time-lagged state labels
            ((frames, features),(frames, weights),(frames,features),(frames, state_labels))

        """
        data = self._seq_trajs(data)
        data_labels = self._seq_trajs(state_labels)

        if data_weights is None: # if no weights are provided, set all weights to 1
            if self._torch_or_numpy == "numpy":
                data_weights = [np.ones_like(data_labels[i]) for i in range(len(data_labels))]
            else:
                data_weights = [torch.ones_like(data_labels[i]) for i in range(len(data_labels))]
        data_weights = self._seq_trajs(data_weights)

        # sanity check of the shape
        print('Checking the shape of the input data and state labels')
        for i in range(len(data)):
            print(f'The shape of the input data is {data[i].shape}, '
                  f'the shape of the state labels is {data_labels[i].shape}',
                  f'the shape of the weights is {data_weights[i].shape}')

            if len(data_labels[i].shape) == 1 or data_labels[i].shape[1] == 1:
                print('The state labels should be one-hot encoded, please check the shape of the state labels')
                print('Now we will convert the state labels to one-hot encoding')
                data_labels[i] = torch.eye(data_labels[i].max() + 1)[data_labels[i]]
                print(f'The shape of the state labels is {data_labels[i].shape}')

            assert data[i].shape[0] == data_labels[i].shape[0] == data_weights[i].shape[0]

        num_trajs = len(data)

        # dataset = []
        instant_data = []
        time_lagged_data = []
        instant_weights = []
        time_lagged_labels = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time
            for i in range(L_re):
                # dataset.append((data[k][i, :], data_weights[k][i], data[k][i + lag_time, :], data_labels[k][i + lag_time]))
                instant_data.append(data[k][i, :])
                time_lagged_data.append(data[k][i + lag_time, :])
                instant_weights.append(data_weights[k][i])
                time_lagged_labels.append(data_labels[k][i + lag_time])

        instant_data = torch.stack(instant_data)
        time_lagged_data = torch.stack(time_lagged_data)
        instant_weights = torch.stack(instant_weights)
        time_lagged_labels = torch.stack(time_lagged_labels)

        dataset = SPIBDataset(instant_data, instant_weights, time_lagged_data, time_lagged_labels)

        return dataset

    def create_boundary_dataset(self, data, ina, inb):
        """
        Create a dataset for the boundary condition.

        This dataset is used for VCN/SVCN prediction.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        ina : list or ndarray
            The initial condition.

        inb : list or ndarray
            The final condition.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has three elements: one is the instantaneous data frame, the other two are the boundary conditions

        """

        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            for i in range(L_all):
                dataset.append((data[k][i, :], ina[k][i], inb[k][i]))

        return dataset

    def create_vcn_dataset(self, data, ina, inb, lag_time):
        """
        Create a dataset for training VCNs.

        This dataset is used for VCN training/validation.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.
        ina : list or ndarray
            The initial condition.
        inb : list or ndarray
            The final condition.
        lag_time : int
            The lag time used to create the dataset.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has 6 elements:
            instantaneous and time-lagged data frames,
            instantaneous and time-lagged initial conditions,
            instantaneous and time-lagged final conditions.

        """
        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []

        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time

            assert ina[k].shape == inb[k].shape == (L_all, 1)

            data_traj = data[k]

            if self._torch_or_numpy == "numpy":
                ina_traj = ina[k].astype(bool)
                inb_traj = inb[k].astype(bool)
            else:
                ina_traj = ina[k].bool()
                inb_traj = inb[k].bool()

            for i in range(L_re):
                dataset.append(
                    (
                        data_traj[i],
                        data_traj[i + lag_time],
                        ina_traj[i],
                        ina_traj[i + lag_time],
                        inb_traj[i],
                        inb_traj[i + lag_time],
                    )
                )

        return dataset

    def create_svcn_dataset(self, data, ina, inb, lag_time):
        """
        Create a dataset for training SVCNs.

        This dataset is used for SVCN training/validation.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.
        ina : list or ndarray
            The initial condition.
        inb : list or ndarray
            The final condition.
        lag_time : int
            The lag time used to create the dataset.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has 12 elements:
            instantaneous and time-lagged data frames,
            instantaneous and time-lagged initial conditions,
            instantaneous and time-lagged final conditions,
            and 6 other entries.

        """
        if self._torch_or_numpy == "numpy":
            from .util import (
                forward_stop,
                backward_stop,
                count_transition_paths_windows,
            )
        else:
            from .util import (
                forward_stop_torch as forward_stop,
                backward_stop_torch as backward_stop,
                count_transition_paths_windows_torch as count_transition_paths_windows,
            )

        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []

        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time

            assert ina[k].shape == inb[k].shape == (L_all, 1)

            data_traj = data[k]

            if self._torch_or_numpy == "numpy":
                ina_traj = ina[k].astype(bool)
                inb_traj = inb[k].astype(bool)
                ind_traj = np.logical_not(np.logical_or(ina_traj, inb_traj))
            else:
                ina_traj = ina[k].bool()
                inb_traj = inb[k].bool()
                ind_traj = torch.logical_not(torch.logical_or(ina_traj, inb_traj))

            r = backward_stop(ind_traj[:, 0])
            s = forward_stop(ind_traj[:, 0])
            ab = count_transition_paths_windows(
                ind_traj[:, 0], ina_traj[:, 0], inb_traj[:, 0], lag_time
            )[:, None]
            ba = count_transition_paths_windows(
                ind_traj[:, 0], inb_traj[:, 0], ina_traj[:, 0], lag_time
            )[:, None]

            for i in range(L_re):
                dataset.append(
                    (
                        data_traj[i],
                        data_traj[i + lag_time],
                        ina_traj[i],
                        ina_traj[i + lag_time],
                        inb_traj[i],
                        inb_traj[i + lag_time],
                        ind_traj[min(i + lag_time, s[i])],
                        ina_traj[min(i + lag_time, s[i])],
                        inb_traj[min(i + lag_time, s[i])],
                        ina_traj[max(i, r[i + lag_time])],
                        inb_traj[max(i, r[i + lag_time])],
                        ab[i] + ba[i],
                    )
                )

        return dataset

    def create_committor_dataset(self, data, ina, inb, drop_first=0, drop_last=0):
        """
        Create a dataset for committor regression.

        This dataset is also useful for testing predicted committors.

        Note that the target will be nan if the previous or next stopping time is not within the trajectory.
        To avoid this, set `drop_first` and `drop_last` to a sufficiently large number.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.
        ina : list or ndarray
            The initial condition.
        inb : list or ndarray
            The final condition.
        drop_first : int, default=0
            Number of frames to drop from the start of each trajectory.
        drop_last : int, default=0
            Number of frames to drop from the last of each trajectory.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has 4 elements: data frame, initial condition, final condition, target.

        """
        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []

        for k in range(num_trajs):
            L_all = data[k].shape[0]

            assert ina[k].shape == inb[k].shape == (L_all, 1)

            data_traj = data[k]

            if self._torch_or_numpy == "numpy":
                from .util import backward_stop, forward_stop

                ina_traj = ina[k].astype(bool)
                inb_traj = inb[k].astype(bool)
                assert not np.any(np.logical_and(ina_traj, inb_traj))

                ind_traj = np.logical_not(np.logical_or(ina_traj, inb_traj))
                r = backward_stop(ind_traj[:, 0])
                s = forward_stop(ind_traj[:, 0])

                target_r = np.full((L_all, 1), np.nan, dtype=np.float32)
                target_s = np.full((L_all, 1), np.nan, dtype=np.float32)

                target_r[r >= 0] = inb_traj[r[r >= 0]]
                target_s[s < L_all] = inb_traj[s[s < L_all]]

            else:
                from .util import backward_stop_torch, forward_stop_torch

                ina_traj = ina[k].bool()
                inb_traj = inb[k].bool()
                assert not torch.any(torch.logical_and(ina_traj, inb_traj))

                ind_traj = torch.logical_not(torch.logical_or(ina_traj, inb_traj))
                r = backward_stop_torch(ind_traj[:, 0])
                s = forward_stop_torch(ind_traj[:, 0])

                target_r = torch.full((L_all, 1), np.nan, dtype=torch.float32)
                target_s = torch.full((L_all, 1), np.nan, dtype=torch.float32)

                target_r[r >= 0] = inb_traj[r[r >= 0]].float()
                target_s[s < L_all] = inb_traj[s[s < L_all]].float()

            for i in range(drop_first, L_all - drop_last):
                dataset.append((data_traj[i], ina_traj[i], inb_traj[i], target_r[i]))
                dataset.append((data_traj[i], ina_traj[i], inb_traj[i], target_s[i]))

        return dataset

    def create_time_lagged_stop_dataset(self, data, ina, inb, lag_time):
        """Create a time-lagged dataset for the boundary condition with stopped time.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        ina : list or ndarray
            The initial condition.

        inb : list or ndarray
            The final condition.

        lag_time : int
            The lag_time used to create the dataset consisting of time-instant and time-lagged data.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            each tuple has six elements: first two represents the instantaneous data frame
            and the corresponding time-lagged data frame.
        """
        assert len(data) == len(ina) == len(inb)

        if self._torch_or_numpy == "numpy":
            from .util import forward_stop

            # squeeze the ina and inb

            data = self._seq_trajs(data)
            ina = self._seq_trajs(ina)
            inb = self._seq_trajs(inb)

            num_trajs = len(data)
            dataset = []

            for k in range(num_trajs):
                L_all = data[k].shape[0]
                L_re = L_all - lag_time

                ina_traj = ina[k].astype(int)
                inb_traj = inb[k].astype(int)

                ind_traj = 1 - ina_traj - inb_traj
                ind_traj = np.squeeze(ind_traj)

                t0 = np.arange(L_re)
                t1 = t0 + lag_time
                ts = np.minimum(t1, forward_stop(ind_traj)[t0])

                data_traj = data[k][t0]
                data_traj_lag = data[k][t1]
                ind_traj_out = ind_traj[t0, np.newaxis]
                ind_traj_lag = ind_traj[t1, np.newaxis]
                ind_traj_stop = ind_traj[ts, np.newaxis]

                assert len(data_traj) == len(data_traj_lag) == len(ind_traj_lag) == len(ind_traj_stop)

                for i in range(L_re):
                    dataset.append((data_traj[i], data_traj_lag[i], ind_traj_stop[i]))

        elif self._torch_or_numpy == "torch":
            from .util import forward_stop_torch as forward_stop

            data = self._seq_trajs(data)
            ina = self._seq_trajs(ina)
            inb = self._seq_trajs(inb)

            num_trajs = len(data)
            dataset = []

            for k in range(num_trajs):
                L_all = data[k].shape[0]
                L_re = L_all - lag_time

                ina_traj = ina[k].int()
                inb_traj = inb[k].int()

                ind_traj = 1 - ina_traj - inb_traj
                ind_traj = ind_traj.squeeze()

                t0 = torch.arange(L_re)
                t1 = t0 + lag_time
                ts = torch.minimum(t1, forward_stop(ind_traj)[t0])

                data_traj = data[k][t0]
                data_traj_lag = data[k][t1]
                ind_traj_out = ind_traj[t0].unsqueeze(1)
                ind_traj_lag = ind_traj[t1].unsqueeze(1)
                ind_traj_stop = ind_traj[ts].unsqueeze(1)

                assert len(data_traj) == len(data_traj_lag) == len(ind_traj_lag) == len(ind_traj_stop)

                for i in range(L_re):
                    dataset.append((data_traj[i], data_traj_lag[i], ind_traj_stop[i]))

        return dataset

    def load_dataset(self, data_path, mmap_mode="r", data_key="arr_0", to_torch=True, sum_token=False):
        """Load the dataset from the file.

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
        assert mmap_mode in ["r", "r+", "w+", "c", None]

        if self._torch_or_numpy == "torch":
            files = os.listdir(data_path)
            files = [os.path.join(data_path, f) for f in files if f.endswith(".pt")]
            files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        else:
            files = os.listdir(data_path)
            files = [os.path.join(data_path, f) for f in files if f.endswith(".npz")]
            files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        data = []
        for file in tqdm(files):
            if self._torch_or_numpy == "torch":
                traj = torch.load(file, map_location="cpu", mmap=mmap_mode).to(self._dtype)
                if sum_token:
                    traj = traj.sum(-3)
                data.append(traj)

            else:
                if to_torch:
                    traj = np.load(file, mmap_mode=mmap_mode)[data_key].astype(self._dtype)
                    traj = torch.tensor(traj).squeeze()
                    if sum_token:
                        traj = traj.sum(-3)
                    data.append(traj)

                else:
                    traj = np.load(file, mmap_mode=mmap_mode)[data_key].astype(self._dtype)
                    if sum_token:
                        traj = traj.sum(-3)
                    data.append(traj)

        if to_torch:
            self._torch_or_numpy = "torch"

        return data

    def load_dataset_folder(self, data_path, mmap_mode="r", data_key="arr_0", to_torch=True, sum_token=False):
        """Load the dataset from the file.

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
        assert mmap_mode in ["r", "r+", "w+", "c", None]

        files = []
        for root, _, filenames in os.walk(data_path):
            if self._torch_or_numpy == "torch":
                files.extend([os.path.join(root, f) for f in filenames if f.endswith(".pt")])
            else:
                files.extend([os.path.join(root, f) for f in filenames if f.endswith(".npz")])

        files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        data = []
        for file in tqdm(files):
            if self._torch_or_numpy == "torch":
                traj = torch.load(file, map_location="cpu", mmap=mmap_mode).to(self._dtype)
                if sum_token:
                    traj = traj.sum(-3)
                data.append(traj)

            else:
                if to_torch:
                    traj = np.load(file, mmap_mode=mmap_mode)[data_key].astype(self._dtype)
                    traj = torch.tensor(traj).squeeze()
                    if sum_token:
                        traj = traj.sum(-3)
                    data.append(traj)

                else:
                    traj = np.load(file, mmap_mode=mmap_mode)[data_key].astype(self._dtype)
                    if sum_token:
                        traj = traj.sum(-3)
                    data.append(traj)

        if to_torch:
            self._torch_or_numpy = "torch"

        return data


class SPIBDataset(Dataset):
    def __init__(self, data, data_weights, time_lagged_data, time_lagged_labels):
        self.data = data
        self.data_weights = data_weights
        self.time_lagged_labels = time_lagged_labels
        self.time_lagged_data = time_lagged_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data_weights[idx], self.time_lagged_labels[idx]

    def update_labels(self, new_labels):
        if not isinstance(new_labels, torch.Tensor):
            new_labels = torch.tensor(new_labels)

        # Update the labels directly in-place
        self.time_lagged_labels[:] = new_labels