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
    t = torch.nonzero(~in_domain, as_tuple=True)[0]
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
    t = torch.nonzero(~in_domain, as_tuple=True)[0]
    t = torch.cat((torch.tensor([-1]), t, torch.tensor([len(in_domain)])))
    return torch.repeat_interleave(t[:-1], torch.diff(t))[1:]
