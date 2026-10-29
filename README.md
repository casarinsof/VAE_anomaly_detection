# Variational autoencoder for anomaly detection

This repository is a personal fork of [Michedev/VAE_anomaly_detection](https://github.com/Michedev/VAE_anomaly_detection), the work of the original author, Michedev. It contains a variational autoencoder (VAE) for anomaly detection following
[Variational Autoencoder based Anomaly Detection using Reconstruction Probability (An & Cho)](https://www.semanticscholar.org/paper/Variational-Autoencoder-based-Anomaly-Detection-An-Cho/061146b1d7938d7a8dae70e3531a00fceb3c78e8).
The decoder's last layer is replaced by two layers that estimate the mean and variance of p(x|z), so anomalies can be scored by reconstruction probability.

## What this fork changes

Compared with `upstream/master`, this fork contains no original code changes: it is a snapshot of an earlier upstream state, kept for personal experiments. The only modifications are the layout and housekeeping done in this branch (package + `scripts/`, `.gitignore`, README). Upstream has since moved on (e.g. packaging, tests, CI), which this fork does not include.

## Structure

```
vae_anomaly/
  vae.py         PyTorch model (VAEAnomaly)
  vae_tf1.py     original TensorFlow 1 implementation
  dataset.py     load_dataset(): replace with your own dataset
scripts/
  train.py       training entry point (pytorch-ignite)
requirements.txt
```

## Installation

```
pip install -r requirements.txt
```

`vae_tf1.py` additionally needs `tensorflow` (1.x), `numpy` and `scipy`.

## Training

Edit `load_dataset()` in `vae_anomaly/dataset.py` (it returns random data by default), adjust the encoder/decoder in `vae_anomaly/vae.py` if needed, then from the repository root:

```
python -m scripts.train --input-size <N>   # add -h for all options
tensorboard --logdir=run
```

The model, parameters and TensorBoard logs are stored in `run/<id>/`.

## Predicting

```python
import torch
from vae_anomaly.vae import VAEAnomaly

model = VAEAnomaly(input_size=50, latent_size=32)  # or read them from run/0/train_config.yaml
model.load_state_dict(torch.load('run/0/model.pth'))
outliers = model.is_anomaly(X_test)
```

## Credits

Original implementation by [Michedev](https://github.com/Michedev/VAE_anomaly_detection). Refer to the upstream repository for its license.
