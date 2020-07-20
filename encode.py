import os
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

snapshots_path = os.path.join(os.getcwd(), "snapshots")
result_folder = "2020-07-01_08:53:32_vortices_conv3dhvae_2level_vamppriorK_500_wu100_z1_40_z2_40"
results_path = os.path.join(snapshots_path, result_folder)


model = torch.load(os.path.join(results_path, "conv3dhvae_2level.model"))
args = torch.load(os.path.join(results_path, "conv3dhvae_2level.config"))
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

from vae_vampprior.datasets.vortices.dataset import VoxelDataset

dataset = VoxelDataset(os.path.join('datasets', 'vortices', 'data'), data_only=False)

loader = data_utils.DataLoader(dataset, batch_size=90, shuffle=True, **kwargs)

model = model.cuda()

latent_lists = list()
vortids = list()

model.eval()
with torch.no_grad():

    for batch, batchids in tqdm(loader):
        _, _, z1_q, _, _, z2_q, _, _, _, _ = model(batch.cuda())
        latent_vector = torch.cat([z1_q, z2_q], dim=-1)

        latent_lists.append(latent_vector.cpu())
        vortids.append(batchids.cpu())

latent_vectors = torch.cat(latent_lists)
vortex_ids = torch.cat(batchids)



print("Finished")