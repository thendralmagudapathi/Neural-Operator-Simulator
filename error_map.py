# evaluate_using_error_heat_map

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from fno_train import FNO2d

# model load from fno.py
def load_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    model = FNO2d(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model

# data loading for testing
def load_data(path):
    arr = np.load(path)
    inp, out = arr['a'], arr['u']
    inp = inp[..., None].astype(np.float32)
    out = out[..., None].astype(np.float32)
    return TensorDataset(torch.from_numpy(inp), torch.from_numpy(out))

# Heat maps plot
def eval_maps(model, test_path, output_dir):
    ds = load_data(test_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    device = next(model.parameters()).device

    for i, (a, true_u) in enumerate(loader):
        a = a.to(device)
        pred_u = model(a).cpu().numpy()[0,...,0]
        err = np.abs(pred_u - true_u.numpy()[0,...,0])
        # plot
        plt.figure(figsize=(6,5))
        plt.imshow(err, cmap='hot')
        plt.colorbar(label='Absolute error')
        plt.title(f'Sample {i} error map, mean error = {err.mean():.4f}')
        plt.savefig(f"{output_dir}/error_map_{i:03d}.png")
        plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test', type=str, default='data/test.npz')
    parser.add_argument('--outdir', type=str, default='error_maps')
    args = parser.parse_args()

    model = load_model(args.model)
    eval_maps(model, args.test, args.outdir)
