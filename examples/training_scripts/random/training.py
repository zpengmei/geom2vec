import argparse
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from geom2vec.data import Preprocessing
from geom2vec import Lobe
from geom2vec.downstream_models import VAMPNet
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a VAMPNet model on trpcage')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model')
parser.add_argument('--training_fraction', type=float, default=0.8, help='Fraction of the data to use for training')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--mixer', type=str, default='mlp', help='The token mixer to use', choices=['mlp', 'sf', 'sm'])
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
parser.add_argument('--lag_time', type=int, default=25, help='Lag time for time-lagged dataset')
parser.add_argument('--system', type=str, default='trpcage', help='System to train on')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels for mixer')
parser.add_argument('--intermediate_channels', type=int, default=256, help='Intermediate channels for mixer')
parser.add_argument('--output_channels', type=int, default=5, help='Output channels of the network')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in final MLP')
parser.add_argument('--batch_norm', action='store_true', help='Whether to use batch normalization', default=True)
parser.add_argument('--vector_feature', action='store_true', help='Whether to use vector features')
parser.add_argument('--mlp_dropout', type=float, default=0.2, help='Dropout for MLP')
parser.add_argument('--mlp_out_activation', type=str, default=None, help='Output activation for MLP')
parser.add_argument('--num_mixer_layers', type=int, default=4, help='Number of mixer layers')
parser.add_argument('--pooling', type=str, default='cls', help='Pooling method for mixer', choices=['cls', 'mean', 'sum'])
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout for mixer')
parser.add_argument('--num_tokens', type=int, default=10, help='Number of tokens')
parser.add_argument('--token_dim', type=int, default=64, help='Token dimension for submixer')
parser.add_argument('--attn_map', action='store_true', help='Whether to use attention map', default=True)
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
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

# Load and preprocess dataset
folder_paths = {
    'trpcage': f'/project/dinner/zpengmei/prots/trpcage/features_{args.hidden_channels}/vis_vs',
    'chignolin': '/project/dinner/zpengmei/prots/chignolin/features/vis_vs',
    'villin': '/project/dinner/zpengmei/prots/villin/features_516'
}

folder_path = folder_paths.get(args.system)
if folder_path is None:
    raise ValueError('Invalid system specified')

preprocess = Preprocessing(torch_or_numpy='numpy')
raw_dataset = preprocess.load_dataset(data_path=folder_path, mmap_mode='r', data_key='features_masked', to_torch=True)
dataset = torch.cat(raw_dataset, dim=0).to(torch.float32)

if args.mixer == 'mlp':
    dataset = dataset.sum(dim=1)
del raw_dataset

time_lagged_dataset = preprocess.create_time_lagged_dataset(dataset.to(torch.float32), lag_time=args.lag_time)

# Split the dataset into training and validation sets
training_fraction = args.training_fraction
train_data, val_data = torch.utils.data.random_split(
    time_lagged_dataset, [int(training_fraction * len(time_lagged_dataset)), len(time_lagged_dataset) - int(training_fraction * len(time_lagged_dataset))]
)

print('train_data:', len(train_data))
print('val_data:', len(val_data))

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

# Define the device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Initialize the model
if args.mixer == 'mlp':
    net = Lobe(
        hidden_channels=args.hidden_channels,
        intermediate_channels=args.intermediate_channels,
        output_channels=args.output_channels,
        num_layers=args.num_layers,
        batch_norm=args.batch_norm,
        vector_feature=args.vector_feature,
        mlp_dropout=args.mlp_dropout,
        mlp_out_activation=args.mlp_out_activation,
        device=device,
    )
elif args.mixer == 'sf':
    net = Lobe(
        hidden_channels=args.hidden_channels,
        intermediate_channels=args.intermediate_channels,
        output_channels=args.output_channels,
        num_layers=args.num_layers,
        batch_norm=args.batch_norm,
        vector_feature=args.vector_feature,
        mlp_dropout=args.mlp_dropout,
        mlp_out_activation=args.mlp_out_activation,
        device=device,
        token_mixer='subformer',
        num_mixer_layers=args.num_mixer_layers,
        pooling=args.pooling,
        dropout=args.dropout,
        num_tokens=args.num_tokens,
        token_dim=args.token_dim,
        attn_map=args.attn_map
    )
elif args.mixer == 'sm':
    net = Lobe(
        hidden_channels=args.hidden_channels,
        intermediate_channels=args.intermediate_channels,
        output_channels=args.output_channels,
        num_layers=args.num_layers,
        batch_norm=args.batch_norm,
        vector_feature=args.vector_feature,
        mlp_dropout=args.mlp_dropout,
        mlp_out_activation=args.mlp_out_activation,
        device=device,
        token_mixer='submixer',
        num_mixer_layers=args.num_mixer_layers,
        pooling=args.pooling,
        dropout=args.dropout,
        num_tokens=args.num_tokens,
        token_dim=args.token_dim,
        attn_map=args.attn_map
    )
else:
    raise ValueError('Invalid mixer type')

model = VAMPNet(
    lobe=net,
    learning_rate=args.learning_rate,
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
    train_valid_interval=args.train_valid_interval,
    valid_patience=args.valid_patience
)
end = time.time()

print('Time elapsed:', end - start)

# Save the model
save_path = os.path.join(args.save_dir, f"mixer_{args.mixer}_seed_{args.seed}_training_fraction_{args.training_fraction}")
if not os.path.exists(save_path):
    os.makedirs(save_path)
model.save_model(save_path)

# Transform the dataset and save collective variables (CVs)
cvs = model.transform(dataset, return_cv=True, lag_time=args.lag_time, batch_size=args.batch_size)
torch.save(cvs, os.path.join(save_path, 'cvs.pt'))

# Save training time and GPU information
with open(os.path.join(save_path, 'time.txt'), 'w') as f:
    f.write(str(end - start))
    if torch.cuda.is_available():
        f.write('\n' + torch.cuda.get_device_name(0))
