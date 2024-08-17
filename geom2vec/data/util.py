import numpy as np
import torch


def forward_stop(in_domain):
    """
    Find the first exit time from the domain.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) ndarray of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(d).

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    t = np.concatenate([[-1], t, [len(in_domain)]])
    return np.repeat(t[1:], np.diff(t))[:-1]


def backward_stop(in_domain):
    """
    Find the last entry time into the domain.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) ndarray of int
        Last entry time into the domain for trajectories starting at
        each frame of the input trajectory. A last entry time not within
        the trajectory is indicated by -1.

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    t = np.concatenate([[-1], t, [len(in_domain)]])
    return np.repeat(t[:-1], np.diff(t))[1:]


def count_transition_paths(in_domain, in_reactant, in_product):
    """
    Count the number of complete transition paths within the trajectory.

    Parameters
    ----------
    in_domain : (n,) ndarray of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) ndarray of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) ndarray of {bool, int}
        Whether each frame is in the product.

    Returns
    -------
    int
        Number of complete transition paths.

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    return np.sum(in_reactant[t[:-1]] * in_product[t[1:]])


def count_transition_paths_windows(in_domain, in_reactant, in_product, lag_time):
    """
    Count the number of complete transition paths within each window.

    Parameters
    ----------
    in_domain : (n,) ndarray of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) ndarray of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) ndarray of {bool, int}
        Whether each frame is in the product.
    lag_time : int
        Lag time in frames. Each window contains ``lag_time + 1`` frames.

    Returns
    -------
    (n - lag,) ndarray of int
        Number of complete transition paths within each window.

    """
    (t,) = np.nonzero(np.logical_not(in_domain))

    is_transition_path = np.logical_and(in_reactant[t[:-1]], in_product[t[1:]])

    initial_count = np.zeros(len(in_domain), dtype=int)
    initial_count[t[:-1][is_transition_path]] = 1
    initial_count = np.concatenate([[0], np.cumsum(initial_count)])

    final_count = np.zeros(len(in_domain), dtype=int)
    final_count[t[1:][is_transition_path]] = 1
    final_count = np.concatenate([[0], np.cumsum(final_count)])

    out = final_count[lag_time + 1 :] - initial_count[: -(lag_time + 1)]
    out = np.maximum(out, 0)
    return out


def forward_stop_torch(in_domain):
    """
    Find the first exit time from the domain.

    Parameters
    ----------
    in_domain : (n,) tensor of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) tensor of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(d).

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)
    t = torch.cat((torch.tensor([-1]), t, torch.tensor([len(in_domain)])))
    return torch.repeat_interleave(t[1:], torch.diff(t))[:-1]


def backward_stop_torch(in_domain):
    """
    Find the last entry time into the domain.

    Parameters
    ----------
    in_domain : (n,) tensor of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) tensor of int
        Last entry time into the domain for trajectories starting at
        each frame of the input trajectory. A last entry time not within
        the trajectory is indicated by -1.

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)
    t = torch.cat((torch.tensor([-1]), t, torch.tensor([len(in_domain)])))
    return torch.repeat_interleave(t[:-1], torch.diff(t))[1:]


def count_transition_paths_windows_torch(in_domain, in_reactant, in_product, lag_time):
    """
    Count the number of complete transition paths within each window.

    Parameters
    ----------
    in_domain : (n,) tensor of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) tensor of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) tensor of {bool, int}
        Whether each frame is in the product.
    lag_time : int
        Lag time in frames. Each window contains ``lag_time + 1`` frames.

    Returns
    -------
    (n - lag,) tensor of int
        Number of complete transition paths within each window.

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)

    is_transition_path = torch.logical_and(in_reactant[t[:-1]], in_product[t[1:]])

    initial_count = torch.zeros(len(in_domain), dtype=torch.int)
    initial_count[t[:-1][is_transition_path]] = 1
    initial_count = torch.cat(
        [torch.zeros(1, dtype=torch.int), torch.cumsum(initial_count, dim=0)]
    )

    final_count = torch.zeros(len(in_domain), dtype=torch.int)
    final_count[t[1:][is_transition_path]] = 1
    final_count = torch.cat(
        [torch.zeros(1, dtype=torch.int), torch.cumsum(final_count, dim=0)]
    )

    out = final_count[lag_time:] - initial_count[:-lag_time]
    out = torch.maximum(out, torch.tensor(0, dtype=torch.int))
    return out


def count_transition_paths_torch(in_domain, in_reactant, in_product):
    """
    Count the number of complete transition paths within the trajectory.

    Parameters
    ----------
    in_domain : (n,) tensor of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) tensor of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) tensor of {bool, int}
        Whether each frame is in the product.

    Returns
    -------
    int
        Number of complete transition paths.

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)
    return torch.sum(in_reactant[t[:-1]] * in_product[t[1:]])
