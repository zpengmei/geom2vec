import torch
import numpy as np
import os
from tqdm import tqdm


class Preprocessing:
    """Preprocess the original trajectories to create datasets for training.

    Parameters
    ----------
    dtype : dtype, default = float
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
                    data[i] = torch.tensor(data[i])
                else:
                    data[i] = data[i].clone().detach()

        return data

    def create_time_lagged_dataset(self, data, lag_time):
        """Create a time-lagged dataset.

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

    def create_boundary_dataset(self, data, ina, inb):
        """Create a dataset for the boundary condition.

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

            ind = 1 - ina - inb
            data = self._seq_trajs(data)
            ind = self._seq_trajs(ind)

            num_trajs = len(data)
            dataset = []

            for k in range(num_trajs):
                L_all = data[k].shape[0]
                L_re = L_all - lag_time

                t0 = np.arange(L_re)
                t1 = t0 + lag_time
                ts = np.minimum(t1, forward_stop(ind[k])[t0])

                data_traj = data[k][t0]
                data_traj_lag = data[k][t1]
                ind = ind[k][t0, np.newaxis]
                ind_lag = ind[k][t1, np.newaxis]
                ind_all = ind[k][ts, np.newaxis]

                assert len(data_traj) == len(data_traj_lag) == len(ind_lag) == len(ind_all)

                for i in range(L_re):
                    dataset.append((data_traj[i], data_traj_lag[i], ind[i], ind_lag[i], ind_all[i]))

        elif self._torch_or_numpy == "torch":
            from .util_torch import forward_stop

            if ina.dtype == torch.bool:
                ina = ina.long()
                inb = inb.long()

            ind = 1 - ina - inb
            data = self._seq_trajs(data)
            ind = self._seq_trajs(ind)

            num_trajs = len(data)
            dataset = []

            for k in range(num_trajs):
                L_all = data[k].shape[0]
                L_re = L_all - lag_time

                t0 = torch.arange(L_re)
                t1 = t0 + lag_time
                ts = torch.minimum(t1, forward_stop(ind[k])[t0])

                data_traj = data[k][t0]
                data_traj_lag = data[k][t1]
                ind = ind[k][t0, None]
                ind_lag = ind[k][t1, None]
                ind_all = ind[k][ts, None]

                assert len(data_traj) == len(data_traj_lag) == len(ind_lag) == len(ind_all)

                for i in range(L_re):
                    dataset.append((data_traj[i], data_traj_lag[i], ind[i], ind_lag[i], ind_all[i]))

        return dataset

    def load_dataset(self, data_path, mmap_mode="r"):
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

        # iterate over the .npz files and load the data

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
                data.append(
                    torch.load(file, map_location="cpu", mmap=mmap_mode).to(self._dtype)
                )
            else:
                data.append(
                    np.load(file, mmap_mode=mmap_mode)["arr_0"].astype(self._dtype)
                )

        return data
