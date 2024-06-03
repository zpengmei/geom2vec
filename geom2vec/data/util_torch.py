import torch


def forward_stop(in_domain):
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


def backward_stop(in_domain):
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
