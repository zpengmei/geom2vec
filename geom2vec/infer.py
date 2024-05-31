import torch
import numpy as np
from typing import List
from tqdm import tqdm
from torch_scatter import scatter
import os


def create_model(model_type,
                 checkpoint_path,
                 cutoff=7.5,
                 hidden_channels=128,
                 num_layers=6,
                 num_rbf=64,
                 device='cuda'):
    assert model_type in ['et', 'vis', 'tn']  # only support ET, ViSNet, and TensorNet

    model = None

    if model_type == 'et':
        from geom2vec.representation_models.torchmd.main_model import get_args, create_model
        args = get_args(hidden_channels, num_layers, num_rbf, num_heads=8, cutoff=cutoff, rep_model='et')
        model = create_model(args)
    elif model_type == 'vis':
        from geom2vec.representation_models.visnet import ViSNet
        model = ViSNet(hidden_channels=hidden_channels, cutoff=cutoff, num_rbf=num_rbf, vecnorm_type='max_min',
                       trainable_vecnorm=True)
    elif model_type == 'tn':
        from geom2vec.representation_models.torchmd.main_model import get_args, create_model
        args = get_args(hidden_channels, num_layers, num_rbf, num_heads=8, cutoff=cutoff, rep_model='tensornet')
        model = create_model(args)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    return model.to(device)


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
        torch_or_numpy: str = 'torch',
        file_name_list: List[str] = None
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
    - torch_or_numpy: str, default = 'torch'
        The type of data to be stored in the output file.
    - file_name_list: list of str, default = None
        The list of file names to be saved.

    """
    model.eval()
    model.to(device=device)

    num_atoms = len(atomic_numbers)

    cg_map = None
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
            traj = torch.from_numpy(traj).float().to(device)
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

                    cg_rep.detach().cpu()
                    out_list.append(cg_rep)
                    continue

                atom_rep = atom_rep.detach().cpu()
                out_list.append(atom_rep.sum(dim=1))
                torch.cuda.empty_cache()

            # concatenate the output batches
            traj_rep = torch.cat(out_list, dim=0)
            if torch_or_numpy == 'numpy':
                traj_rep = traj_rep.numpy()
                saving_file_name = file_name_list[i] if file_name_list is not None else f"traj_{i}"
                saving_path = os.path.join(saving_path, f"{saving_file_name}.npz")
                np.savez(saving_path, traj_rep)
                print(f"Trajectory {i} has been saved to {saving_path} using numpy.")
            elif torch_or_numpy == 'torch':
                saving_file_name = file_name_list[i] if file_name_list is not None else f"traj_{i}"
                saving_path = os.path.join(saving_path, f"{saving_file_name}.pt")
                torch.save(traj_rep, saving_path)
                print(f"Trajectory {i} has been saved to {saving_path} using torch.")
            else:
                print("Invalid option for torch_or_numpy. Please choose either 'torch' or 'numpy'.")

    return None
