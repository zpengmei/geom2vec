import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from torch_scatter import scatter


def infer_traj(
        model: torch.nn.Module,
        hidden_channels: int,
        data: List[np.ndarray],
        atomic_numbers: np.ndarray,
        device: torch.device,
        saving_path: str,
        batch_size: int = 32,
        cg_mapping: np.ndarray = None,
        atom_mask: np.ndarray = None,
        cg_mask: np.ndarray = None,
):
    r"""
    This function is used to infer the trajectories of data.

    Input:
    - model: torch.nn.Module
        The model to be used for inference.
    - data: list of ndarray
        The trajectories of data in xyz.
    - atomic_numbers: ndarray
        The atomic numbers of the atoms.
    - device: torch.device
        The device on which the torch modules are executed.
    - saving_path: str
        The path to save the inferred data.
    - cg_mapping: ndarray, default = None
        The coarse-grained mapping to be used.
    - atom_mask: ndarray, default = None
        The mask of the atoms for the inference to save the data.
    - cg_mask: ndarray, default = None
        The mask of the coarse-grained beads for the inference to save the data.
    """
    model.eval()
    model.to(device=device)

    num_atoms = len(atomic_numbers)

    if cg_mapping is not None:
        assert sum(cg_mapping) == num_atoms
        # The sum of the coarse-grained mapping should be equal to the number of atoms
        cg_mapping = torch.from_numpy(cg_mapping).to(device)
        cg_map = torch.repeat_interleave(torch.arange(cg_mapping.shape[0], device=device), cg_mapping, dim=0)

    z = torch.from_numpy(atomic_numbers).to(device)
    with torch.no_grad():
        for i in range(len(data)):
            traj = data[i]

            out_list = []
            for pos_batch in tqdm(torch.split(traj, batch_size, dim=0)):
                n_samples, n_atoms, _ = pos_batch.shape
                z_batch = z.expand(n_samples, -1).reshape(-1).to(device)
                batch_batch = (
                    torch.arange(n_samples).unsqueeze(1).expand(-1, n_atoms).reshape(-1)
                ).to(device)
                x_rep, v_rep, _ = model(z=z_batch, pos=pos_batch.reshape(-1, 3).contiguous().to(device),
                                        batch=batch_batch)
                # Move the data to CPU and append to the output list
                x_rep = x_rep.reshape(-1, num_atoms, 1, hidden_channels)
                v_rep = v_rep.reshape(-1, num_atoms, 3, hidden_channels)
                atom_rep = torch.cat([x_rep, v_rep], dim=-2)

                if atom_mask is not None:
                    atom_rep = atom_rep[:, atom_mask:, :]

                if cg_mapping is not None:
                    cg_rep = scatter(atom_rep, cg_map, dim=1, reduce='add')
                    if cg_mask is not None:
                        cg_rep = cg_rep[:, cg_mask, :, ]

                    cg_rep.detach().cpu().numpy()
                    out_list.append(cg_rep)
                    continue

                atom_rep = atom_rep.detach().cpu().numpy()
                out_list.append(atom_rep.sum(dim=1))
                torch.cuda.empty_cache()

        # concatenate the output batches
        traj_rep = np.concatenate(out_list, axis=0)
        np.savez(saving_path, traj_rep)
        print(f"Trajectory {i} has been saved to {saving_path}.")

    return None

