import random
import os
import numpy as np
import torch
import scipy
from six import string_types
from collections import Counter
import MDAnalysis as mda

mass_mapping = {'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'H': 1.008, 'S': 32.06}
atomic_mapping = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16}


def count_segments(numbers):
    segments = []
    current_segment = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1]:
            current_segment.append(numbers[i])
        else:
            segments.append(current_segment)
            current_segment = [numbers[i]]
    segments.append(current_segment)  # Add the last segment

    # Count elements in each segment
    segment_counts = [Counter(segment) for segment in segments]
    segment_counts_array = np.array([list(segment_count.values()) for segment_count in segment_counts])
    segment_counts_array = np.concatenate(segment_counts_array, axis=0)
    return segment_counts_array


def extract_mda_info(protein, stride=1):
    # input: MDA Universe object with selection, output: positions, atomic_numbers, segment_counts
    # protein = u.select_atoms('protein and prop mass > 1.5 ')

    # Get all residues in the protein selection
    protein_residues = protein.residues
    atomic_masses = protein_residues.masses
    atomic_masses = np.round(atomic_masses, 3)

    atomic_types = [list(mass_mapping.keys())[list(mass_mapping.values()).index(mass)] for mass in atomic_masses]
    atomic_numbers = [atomic_mapping[atom] for atom in atomic_types]

    positions = []
    for ts in protein.trajectory:
        positions.append(protein_residues.positions.copy())

    positions = np.array(positions)[::stride]

    segment_counts = count_segments(protein_residues.resids)

    return positions, np.array(atomic_numbers), np.array(segment_counts)


def extract_mda_info_folder(folder, top_file,stride=1):
    r"""
    do the extraction for all the files in the folder

    Args:
    - folder: str
        The folder containing the .dcd files
    - top_file: str
        The topology file
    - stride: int, default = 1
        The stride to use when extracting the data

    Returns:
    - position_list: list
        The list of positions for each trajectory
    - atomic_numbers: ndarray
        The atomic numbers of the atoms
    - segment_counts: ndarray
        The number counts for each segment
    - dcd_files: list
        The list of dcd files in the folder (important for the order of the trajectories)
    """

    # Get all the .dcd files in the folder
    dcd_files = [f for f in os.listdir(folder) if f.endswith('.dcd')]
    dcd_files.sort()

    position_list = []
    for traj in dcd_files:
        print(f'Processing {traj}')
        u = mda.Universe(top_file, os.path.join(folder, traj))
        positions, atomic_numbers, segment_counts = extract_mda_info(u, stride=stride)
        position_list.append(positions)

    return position_list, atomic_numbers, segment_counts, dcd_files



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def empirical_correlation(x, y):
    x_remove_mean = x - x.mean()
    y_remove_mean = y - y.mean()
    corr = np.mean(x_remove_mean * y_remove_mean) / (
            np.sqrt(np.mean(x_remove_mean * x_remove_mean)) * np.sqrt(np.mean(y_remove_mean * y_remove_mean)))
    return np.abs(corr)


def mean_error_bar(data, confidence=0.95):
    mean = np.mean(data, axis=0)
    down, up = scipy.stats.t.interval(confidence=confidence, df=len(data) - 1, loc=np.mean(data, axis=0),
                                      scale=scipy.stats.sem(data, axis=0))
    return mean, down, up


def rao_blackwell_ledoit_wolf(S, n):
    """ Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance matrix.

    Parameters
    ----------
    S : array, shape=(n, n)
        Sample covariance matrix (e.g. estimated with np.cov(X.T))

    n : int
        Number of data points.

    Returns
    -------
    sigma : array, shape=(n, n)

    References
    ----------
    .. [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
        estimation of high dimensional covariance matrices" ICASSP (2009)
    """

    p = len(S)
    assert S.shape == (p, p)

    alpha = (n - 2) / (n * (n + 2))
    beta = ((p + 1) * n - 2) / (n * (n + 2))

    trace_S2 = np.sum(S * S)  # np.trace(S.dot(S))
    U = ((p * trace_S2 / np.trace(S) ** 2) - 1)
    rho = min(alpha + beta / U, 1)

    F = (np.trace(S) / p) * np.eye(p)

    return (1 - rho) * S + rho * F, rho


class ContourPlot2D:

    def __init__(self, bw_method='scotts', num_grids=120, cut=3, clip=None,
                 temperature=310., shade=True, alpha=0.6, vmin=0, vmax=7, n_levels=15):

        self._bw_method = bw_method
        self._num_grids = num_grids
        self._cut = cut
        if clip is None:
            self._clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        self._temperature = temperature
        self._shade = shade
        self._alpha = alpha
        self._vmin = vmin
        self._vmax = vmax
        self._n_levels = n_levels

    def _kde_support(self, data, bw, num_grids, cut, clip):

        support_min = max(data.min() - bw * cut, clip[0])
        support_max = min(data.max() + bw * cut, clip[1])

        return np.linspace(support_min, support_max, num_grids)

    def _scipy_bivariate_kde(self, data, bw_method, num_grids, cut, clip):

        from scipy import stats
        kde = stats.gaussian_kde(data.T)
        std = data.std(axis=0, ddof=1)

        if isinstance(bw_method, string_types):
            bw_x = getattr(kde, "%s_factor" % bw_method)() * std[0]
            bw_y = getattr(kde, "%s_factor" % bw_method)() * std[1]
        else:
            raise ValueError('Please input the string of a valid bandwidth method.')

        x_support = self._kde_support(data[:, 0], bw_x, num_grids, cut, clip[0])
        y_support = self._kde_support(data[:, 1], bw_y, num_grids, cut, clip[1])

        xx, yy = np.meshgrid(x_support, y_support)
        z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)

        return xx, yy, z

    def _thermo_transform(self, z, temperature):

        from scipy.constants import Avogadro, Boltzmann, calorie_th
        THERMO_CONSTANT = 10 ** -3 * Boltzmann * Avogadro / calorie_th

        return - THERMO_CONSTANT * temperature * np.log(z)

    def plot(self, data, ax=None, cbar=True, cbar_kwargs={},
             xlabel=None, ylabel=None, labelsize=10):

        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()

        X, Y, Z = self._scipy_bivariate_kde(data, self._bw_method, self._num_grids, self._cut, self._clip)
        Z = self._thermo_transform(Z, self._temperature)

        if self._vmin is None:
            self._vmin = -1E-12
        if self._vmax is None:
            self._vmax = np.percentile(Z, 50)

        if self._shade:
            cf = ax.contourf(X, Y, Z - Z.min(), levels=np.linspace(self._vmin, self._vmax, self._n_levels),
                             alpha=self._alpha, zorder=1, vmin=self._vmin, vmax=self._vmax)

        cs = ax.contour(X, Y, Z - Z.min(), cmap=plt.get_cmap('bone_r'),
                        levels=np.linspace(self._vmin, self._vmax, self._n_levels), alpha=1,
                        zorder=1, vmin=self._vmin, vmax=self._vmax)

        if cbar:
            if self._shade:
                cbar = plt.colorbar(cf, **cbar_kwargs)
            else:
                cbar = plt.colorbar(cs, **cbar_kwargs)

            cbar.ax.tick_params(labelsize=labelsize)
            cbar.set_label('Free energy (kcal/mol)', fontsize=labelsize)

        ax.grid(zorder=0)

        if xlabel:
            ax.set_xlabel(xlabel, size=labelsize)
        if ylabel:
            ax.set_ylabel(ylabel, size=labelsize)

        return ax
