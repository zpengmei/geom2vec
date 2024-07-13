import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import mdtraj as md
import MDAnalysis as mda
from geom2vec.data import Preprocessing
from geom2vec.downstream_models.lobe_otf import Lobe
from geom2vec.downstream_models import VAMPNet
from geom2vec import create_model

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a VAMPNet model on trpcage')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model')
parser.add_argument('--training_fraction', type=float, default=0.8, help='Fraction of the data to use for training')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
parser.add_argument('--system', type=str, default='chignolin', help='System to train on')
parser.add_argument('--batch_size', type=int, default=250, help='Batch size for training')
parser.add_argument('--batch_norm', action='store_true', help='Whether to use batch normalization')
parser.add_argument('--vector_feature', action='store_true', help='Whether to use vector features')
parser.add_argument('--mlp_dropout', type=float, default=0.2, help='Dropout for MLP')
parser.add_argument('--mlp_out_activation', type=str, default=None, help='Output activation for MLP')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
parser.add_argument('--grad_accum_steps', type=int, default=4, help='Number of gradient accumulation steps')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use for training')
parser.add_argument('--train_patience', type=int, default=1000, help='Patience for training')
parser.add_argument('--train_valid_interval', type=int, default=100, help='Interval between training and validation')
parser.add_argument('--valid_patience', type=int, default=10, help='Patience for validation')
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train for')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.system == 'trpcage':
    # Load trajectory data using MDTraj
    num_atoms = 272
    lag_time = 25
    output_channels = 5
    topology_file = "/project/dinner/anton_data/TRP_cage/trpcage.pdb"
    trajectory_files = [f"/project/dinner/anton_data/TRP_cage/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein-{i:03d}.dcd" for i in range(0, 105)]

    traj = md.load(trajectory_files[0], top=topology_file)
    for traj_file in tqdm(trajectory_files[1:]):
        traj = md.join([traj, md.load(traj_file, top=topology_file)])


elif args.system == 'chignolin':

    num_atoms = 166
    lag_time = 25
    output_channels = 4
    topology_file = "/project/dinner/anton_trajs/Chignolin/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/chignolin.pdb"
    trajectory_file = "/project/dinner/anton_trajs/Chignolin/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/CLN025-0-protein-000.dcd"

    for i in tqdm(range(0, 54)):
        traj_file = f"/project/dinner/anton_trajs/Chignolin/DESRES-Trajectory_CLN025-0-protein/CLN025-0-protein/CLN025-0-protein-{i:03d}.dcd"
        if i == 0:
            traj = md.load(traj_file, top=topology_file)
        else:
            traj = md.join([traj, md.load(traj_file, top=topology_file)])

elif args.system == 'villin':

    num_atoms = 577
    lag_time = 50
    output_channels = 4
    topology_file = "/project/dinner/anton_data/DESRES-Trajectory_2F4K-0-protein/villin.pdb"
    trajectory_file = "/project/dinner/anton_data/DESRES-Trajectory_2F4K-0-protein/2F4K-0-protein/2F4K-0-protein-000.dcd"

    for i in tqdm(range(0, 63)):
        traj_file = f"/project/dinner/anton_data/DESRES-Trajectory_2F4K-0-protein/2F4K-0-protein/2F4K-0-protein-{i:03d}.dcd"
        if i == 0:
            traj = md.load(traj_file, top=topology_file)
        else:
            traj = md.join([traj, md.load(traj_file, top=topology_file)])

# Preprocess data
atom_types = [atom.element.symbol for atom in traj.top.atoms]
mask = [atom != 'H' for atom in atom_types]
xyz = torch.tensor(traj.xyz).reshape(-1, num_atoms, 3)
xyz = xyz[:,mask,:]
if args.system != 'chignolin':
    xyz = xyz[::2]

preprocess = Preprocessing(torch_or_numpy='torch')
time_lagged_dataset = preprocess.create_time_lagged_dataset(xyz.to(torch.float32), lag_time=lag_time)

# Split the dataset into training and validation sets
train_data_unsplit, val_data = torch.utils.data.random_split(
    time_lagged_dataset, [int(0.5 * len(time_lagged_dataset)), len(time_lagged_dataset) - int(0.5 * len(time_lagged_dataset))]
)

train_data, _ = torch.utils.data.random_split(
    train_data_unsplit, [int(args.training_fraction * len(train_data_unsplit)), len(train_data_unsplit) - int(args.training_fraction * len(train_data_unsplit))]
)

print('train_data:', len(train_data))
print('val_data:', len(val_data))

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

# Get atomic numbers using MDAnalysis
u = mda.Universe(topology_file, trajectory_file)
protein = u
protein_residues = protein.select_atoms("prop mass > 1.5 ")  # remove hydrogens

mass_mapping = {"C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "H": 1.008, "S": 32.06, "F": 18.998, "Cl": 35.453}
atomic_mapping = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16, "F": 9, "Cl": 17}

atomic_masses = protein_residues.masses
atomic_masses = np.round(atomic_masses, 3)
atomic_types = [list(mass_mapping.keys())[list(mass_mapping.values()).index(mass)] for mass in atomic_masses]
atomic_numbers = [atomic_mapping[atom] for atom in atomic_types]
atomic_numbers = torch.tensor(atomic_numbers)

# Define the device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# device ='cpu'
# Initialize the representation model
rep_model = create_model(
    model_type='vis',
    cutoff=5,
    hidden_channels=16,
    num_layers=6,
    num_rbf=16,
    device=device
)

# print(atomic_numbers.shape)
# print(train_data[0][0].shape)
# print(xyz.shape)
# Initialize the Lobe network
net = Lobe(
    representation_model=rep_model,
    atomic_numbers=atomic_numbers,
    hidden_channels=16,
    intermediate_channels=16,
    output_channels=output_channels,
    num_layers=2,
    batch_norm=args.batch_norm,
    vector_feature=args.vector_feature,
    mlp_dropout=args.mlp_dropout,
    mlp_out_activation=args.mlp_out_activation,
    device=device,
)

# Initialize the VAMPNet model
model = VAMPNet(
    lobe=net,
    learning_rate=args.learning_rate,
    grad_accum_steps=args.grad_accum_steps,
    optimizer=args.optimizer,
    device=device
)

# Train the model
start = time.time()
model.fit(
    train_loader,
    n_epochs=args.n_epochs,
    validation_loader=val_loader,
    progress=tqdm,
    train_patience=args.train_patience,
    train_valid_interval=args.train_valid_interval * args.grad_accum_steps,
    valid_patience=args.valid_patience
)
end = time.time()
print('Time elapsed:', end - start)

# Create directory to save the model if it doesn't exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
model.save_model(args.save_dir)

# Transform the dataset and save collective variables (CVs)
cvs = model.transform(xyz, return_cv=True, lag_time=args.lag_time, batch_size=args.batch_size)
torch.save(cvs, os.path.join(args.save_dir, 'cvs.pt'))

# Save training time and GPU information
with open(os.path.join(args.save_dir, 'time.txt'), 'w') as f:
    f.write(str(end - start))
    if torch.cuda.is_available():
        f.write('\n' + torch.cuda.get_device_name(0))