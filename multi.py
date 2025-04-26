"""
multi_worker_modal_gpu_coordinator_mixed.py
───────────────────────────────────────────
*Exactly one* Modal app that launches four workers, each on **a
different GPU type**, then federated-averages their checkpoints.

GPU map ↓

┌─────────┬──────────────┐
│worker # │  GPU TYPE    │
├─────────┼──────────────┤
│   0     │   "T4"       │
│   1     │   "A10G"     │
│   2     │   "A100"     │
│   3     │   "H100"     │
└─────────┴──────────────┘

Modify the `GPU_TYPES` list if your quota / region differs.
"""

# ──────────────────────────────────────────────────────────────
# 0. Imports & settings
# ──────────────────────────────────────────────────────────────
import pickle
from pathlib import Path
from typing import List

import modal

NUM_WORKERS     = 4
GPU_TYPES       = ["T4", "A10G", "A100", "H100"]   # must be length-4
BATCH_SIZE      = 128
EPOCHS          = 1
STUB_NAME       = "mixed-gpu-mnist"
CHECKPOINT_FILE = "model_mixed.pt"

assert len(GPU_TYPES) == NUM_WORKERS, "GPU_TYPES length must match NUM_WORKERS"

# ──────────────────────────────────────────────────────────────
# 1. Modal boilerplate
# ──────────────────────────────────────────────────────────────
stub  = modal.Stub(STUB_NAME)
image = (
    modal.Image.debian_slim()
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "tqdm")
)

# ──────────────────────────────────────────────────────────────
# 2. Model + utils (same for every worker)
# ──────────────────────────────────────────────────────────────
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def fed_average(states: List[dict]) -> dict:
    out = {}
    for k in states[0]:
        out[k] = sum(sd[k] for sd in states) / len(states)
    return out


# ──────────────────────────────────────────────────────────────
# 3. Dynamically build one worker-function per GPU type
# ──────────────────────────────────────────────────────────────
worker_fns = []

for idx, gpu_spec in enumerate(GPU_TYPES):
    @stub.function(name=f"train_worker_{gpu_spec}",
                   gpu=gpu_spec,
                   image=image,
                   timeout=60 * 40,
                   cpu=4,
                   memory=14)
    def _worker(worker_id: int,
                world_size: int,
                init_state_bytes: bytes) -> bytes:
        # dataset shard
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = datasets.MNIST("/data/mnist",
                            train=True,
                            download=True,
                            transform=tfm)
        idxs = list(range(worker_id, len(ds), world_size))
        dl   = DataLoader(Subset(ds, idxs),
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2)

        device = torch.device("cuda")
        model  = CNN().to(device)
        model.load_state_dict(pickle.loads(init_state_bytes))
        opt    = optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss_fn(model(xb), yb).backward()
                opt.step()

        return pickle.dumps(model.state_dict())

    worker_fns.append(_worker)

# ──────────────────────────────────────────────────────────────
# 4. Local coordinator
# ──────────────────────────────────────────────────────────────
@stub.local_entrypoint()
def main():
    init = pickle.dumps(CNN().state_dict())

    print("🚀 Dispatching 4 heterogeneous GPU jobs:")
    for i, g in enumerate(GPU_TYPES):
        print(f"   • worker {i} → {g}")

    futures = [
        fn.spawn(w, NUM_WORKERS, init)
        for w, fn in enumerate(worker_fns)
    ]

    merged = fed_average([pickle.loads(f.get()) for f in futures])
    out_path = Path(CHECKPOINT_FILE).absolute()
    torch.save(merged, out_path)
    print(f"✅ Averaged model stored at {out_path}")


# ──────────────────────────────────────────────────────────────
# HOW TO RUN
# ──────────────────────────────────────────────────────────────
#   $ modal run multi_worker_modal_gpu_coordinator_mixed.py
#
# Watch the four jobs in Modal’s dashboard; each provisions a
# different GPU class.  When finished the merged checkpoint
# is on your local FS as `model_mixed.pt`.
#
# ──────────────────────────────────────────────────────────────


