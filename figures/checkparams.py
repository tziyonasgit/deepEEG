# checkparams.py
# or EEGPT_classifier
from deepEEG.EEGPT.downstream.age.Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier
import os
import sys
import torch

# 1) Make your repo importable (adjust if your tree is different)
sys.path.append("/home/chntzi001/deepEEG/EEGPT/downstream/age")
use_channels_names = ['FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'C2', 'C4',
                      'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2']


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {total - trainable:,}")


def load_weights(model, ckpt_path):
    # Prefer safer weights_only=True if your torch supports it
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu",
                          weights_only=True)  # torch>=2.4
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")  # fallback

    state = ckpt.get("state_dict", ckpt)
    # Drop classifier head keys if shape mismatch
    drop = [k for k in list(state) if k.startswith(
        "head.") or "classifier" in k or k.startswith("fc.")]
    for k in drop:
        state.pop(k, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded with strict=False | missing: {len(missing)} | unexpected: {len(unexpected)}")


if __name__ == "__main__":
    safe_logdir = "/tmp/eegpt_paramcheck"  # <-- ensure this is a string path
    os.makedirs(safe_logdir, exist_ok=True)

    # Build the model; keep args consistent with your checkpoint/config
    model = EEGPTClassifier(
        num_classes=4,
        embed_dim=512,
        drop_rate=0.0,
        logdir=safe_logdir,   # <-- fixes the makedirs TypeError
        # add other required kwargs here if your class needs them
    )

    # Optional: freeze/unfreeze here if you want to see trainable counts change
    # for p in model.parameters(): p.requires_grad = False
    # for p in model.head.parameters(): p.requires_grad = True

    # Load weights (optional; the param count is the same with/without weights)
    ckpt_path = "/home/chntzi001/deepEEG/EEGPT/downstream/Checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"
    load_weights(model, ckpt_path)

    count_params(model)
