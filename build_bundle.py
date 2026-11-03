"""Run in the existing Colab session with `%run -i build_bundle.py`.
Requires the already-trained variables `net` and `tok` from the notebook.
"""
from pathlib import Path
import json
import shutil
import zipfile
import torch

try:
    _ns = get_ipython().user_ns
except Exception:
    _ns = globals()


def _get(*names, default=None):
    for key in names:
        if key in _ns and _ns[key] is not None:
            return _ns[key]
    return default


net = _get("net")
tok = _get("tok", "tokenizer")
if net is None or tok is None:
    raise RuntimeError(
        "Could not find trained `net` and tokenizer `tok` in this notebook. "
        "Run this with `%run -i /content/banglabert_gat_gcn_package/build_bundle.py` "
        "in the SAME runtime where training finished."
    )

model_name = _get("name", "nm", default=getattr(net.bb.config, "_name_or_path", "csebuetnlp/banglabert"))
config = {
    "model_name": str(model_name),
    "max_len": int(_get("max_len", "ml", default=256)),
    "g_dim": int(_get("g_dim", "gd", default=net.proj.out_features)),
    "heads": int(_get("heads", "nh", default=net.gat1.heads)),
    "window": int(_get("window", "win", default=2)),
    "drop": float(_get("drop", "dp", default=net.drop.p)),
    "num_labels": 2,
    "labels": {"0": "Non-Hate", "1": "Hate"},
}

root = Path(__file__).resolve().parent
net.eval()
state = {k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}
torch.save({"config": config, "state_dict": state}, root / "model.pkl")

# Recreate tokenizer directory from the current trained notebook tokenizer.
tok_dir = root / "tokenizer"
if tok_dir.exists():
    shutil.rmtree(tok_dir)
tok.save_pretrained(tok_dir)
# Keep the small transformer config locally so reloading needs no remote config lookup.
bb_cfg_dir = root / "backbone_config"
if bb_cfg_dir.exists():
    shutil.rmtree(bb_cfg_dir)
net.bb.config.save_pretrained(bb_cfg_dir)
with open(root / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

zip_path = root.parent / "banglabert_gat_gcn.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob("*"):
        if p.is_file() and p.name not in {".DS_Store"} and "__pycache__" not in p.parts:
            zf.write(p, Path(root.name) / p.relative_to(root))

print("Package completed:", zip_path)
print("Contains model.pkl + model.py + explainability.py + tokenizer/ + requirements.txt + README.md")
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception:
    print("Download the ZIP from the Colab Files panel:", zip_path)
