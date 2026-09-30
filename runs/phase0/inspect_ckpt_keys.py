"""Phase 0: list top-level keys of the released DCP checkpoints and compare them with the FastGen Wan-1.3B net."""
import os
from collections import Counter

import torch
from torch.distributed.checkpoint import FileSystemReader

from fastgen.networks.Wan.network import Wan

CKPTS = [
    "checkpoints/wan_t2v_checkpoints_release/dmd2_t2v_vipe/0001000",
    "checkpoints/wan_t2v_checkpoints_release/ours_wan_t2v_vipe/0000040",
]

# Same kwargs as Wan_1_3B_Config, but without downloading/loading weights (meta device, keys/shapes only).
with torch.device("meta"):
    net = Wan(
        model_id_or_local_path=Wan.MODEL_ID_1_3B,
        enable_logvar_linear=False,
        r_timestep=False,
        use_wan_official_sinusoidal=False,
        load_pretrained=False,
    )
net_sd = net.state_dict()
print(f"FastGen Wan_1_3B net: {len(net_sd)} state_dict entries, "
      f"{sum(v.numel() for v in net_sd.values()) / 1e9:.4f} B elements")

for path in CKPTS:
    print("=" * 100)
    print(path)
    for f in sorted(os.listdir(os.path.dirname(path))):
        print("  file:", f)
    md = FileSystemReader(path + ".net_model").read_metadata()
    keys = list(md.state_dict_metadata.keys())
    print(f"  DCP entries: {len(keys)}")
    print("  top-level prefixes:", dict(Counter(k.split(".")[0] for k in keys)))
    tensor_md = {k: v for k, v in md.state_dict_metadata.items() if hasattr(v, "size")}
    n_el = sum(torch.Size(v.size).numel() for v in tensor_md.values())
    dtypes = Counter(str(v.properties.dtype) for v in tensor_md.values())
    print(f"  tensor elements: {n_el / 1e9:.4f} B, dtypes: {dict(dtypes)}")
    hints = [k for k in keys if any(s in k.lower() for s in ("fake", "disc", "optim", "ema", "exp_avg", "step"))]
    print("  keys mentioning fake/disc/optim/ema/exp_avg/step:", hints[:10], "..." if len(hints) > 10 else "")
    missing = sorted(set(net_sd) - set(keys))
    unexpected = sorted(set(keys) - set(net_sd))
    shape_mismatch = [k for k in set(net_sd) & set(tensor_md) if tuple(net_sd[k].shape) != tuple(tensor_md[k].size)]
    print(f"  vs net: missing={len(missing)} {missing[:10]}")
    print(f"          unexpected={len(unexpected)} {unexpected[:10]}")
    print(f"          shape_mismatch={len(shape_mismatch)} {shape_mismatch[:10]}")
    st = torch.load(path + ".pth", map_location="cpu", weights_only=False)
    print("  .pth:", {k: (v if k == "iteration" else type(v).__name__ + f"(len={len(v)})") for k, v in st.items()})
    print("  .pth scheduler keys:", list(st["scheduler"].keys()), "| callbacks keys:", list(st["callbacks"].keys()))
