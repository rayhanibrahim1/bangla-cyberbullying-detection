"""Simple word-level GAT/GCN explanations for the trained HybridNet."""

import re
import unicodedata
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from IPython.display import display, HTML
from normalizer import normalize


def clean_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return normalize(text)


class Explainer:
    """Creates prediction probabilities and approximate word-level evidence."""

    def __init__(self, model, tokenizer, config, device=None):
        self.model = model
        self.tok = tokenizer
        self.cfg = dict(config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.max_len = int(self.cfg.get("max_len", 256))
        self.window = int(self.cfg.get("window", 2))

    def _encode(self, text):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        starts = [a for a, _ in spans]
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       padding=False, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        word_ids = [bisect_right(starts, a) - 1 if end > a else -1
                    for a, end in offsets]
        visible = sorted(set(i for i in word_ids if i >= 0))
        word_map = {word_i: node_i + 1 for node_i, word_i in enumerate(visible)}
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
            "w": [word_map.get(i, 0) for i in word_ids],
            "word_count": len(visible),
        }
        words = [text[a:b] for i, (a, b) in enumerate(spans) if i in set(visible)]
        return item, words, visible

    def _batch(self, items):
        keys = ["input_ids", "attention_mask", "token_type_ids"]
        b = self.tok.pad([{k: row[k] for k in keys} for row in items],
                         padding=True, pad_to_multiple_of=8, return_tensors="pt")
        b["w"] = torch.zeros_like(b["input_ids"])
        counts = torch.tensor([r["word_count"] for r in items])
        n = int(counts.max()) + 1
        for i, row in enumerate(items):
            b["w"][i, :len(row["w"])] = torch.tensor(row["w"], dtype=torch.long)
        ix = torch.arange(n)
        nm = ix[None, :] <= counts[:, None]
        adj = (ix[:, None] - ix[None, :]).abs() <= self.window
        adj[0, :] = True
        adj[:, 0] = True
        b["nm"] = nm
        b["adj"] = adj[None, :, :] & nm[:, :, None] & nm[:, None, :]
        return {k: v.to(self.device) for k, v in b.items()}

    @staticmethod
    def _node_attention(alpha, edge, n):
        # Aggregate outgoing attention by source node; this is a summary, not causal proof.
        src = edge[0]
        val = alpha.mean(dim=-1)
        score = val.new_zeros(n)
        deg = val.new_zeros(n)
        score.scatter_add_(0, src, val)
        deg.scatter_add_(0, src, torch.ones_like(val))
        return score / deg.clamp_min(1)

    def explain(self, text, show=True):
        text = clean_text(text)
        if not text:
            raise ValueError("Please enter a non-empty sentence.")
        item, words, visible_ids = self._encode(text)
        b = self._batch([item])
        self.model.eval()

        with torch.enable_grad():
            out, info = self.model.forward_with_details(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                token_type_ids=b.get("token_type_ids"),
                w=b["w"], nm=b["nm"], adj=b["adj"],
            )
            logits = out["logits"]
            probs = logits.softmax(dim=-1)[0]
            pred = int(probs.argmax().item())
            other = 1 - pred
            margin = logits[0, pred] - logits[0, other]
            grads = torch.autograd.grad(margin, info["branches"], allow_unused=True)

        n = info["num_nodes"]
        count = min(len(words), item["word_count"])
        words = words[:count]
        branch_scores = []
        for act, grad in zip(info["branches"], grads):
            if grad is None:
                s = torch.zeros(n, device=self.device)
            else:
                s = (act * grad).sum(dim=-1).reshape(info["batch_size"], n)[0]
            branch_scores.append(s.detach().cpu().numpy())

        gat_score = branch_scores[0] + branch_scores[2]
        gcn_score = branch_scores[1] + branch_scores[3]

        att1 = self._node_attention(info["alpha1"], info["edge1"], n)
        att2 = self._node_attention(info["alpha2"], info["edge2"], n)
        att = ((att1 + att2) / 2).detach().cpu().numpy()[1:count + 1]
        if len(att) and att.sum() > 0:
            att = att / att.sum() * 100

        # No word-removal/word-impact column, as requested.
        table = pd.DataFrame({
            "Word": words,
            "GAT attention (%)": np.round(att, 2),
            "GAT contribution*": np.round(gat_score[1:count + 1], 4),
            "GCN contribution*": np.round(gcn_score[1:count + 1], 4),
        })
        labels = {0: "Non-Hate", 1: "Hate"}
        result = {
            "prediction": labels[pred],
            "probabilities": {
                "Non-Hate": float(probs[0].detach().cpu()),
                "Hate": float(probs[1].detach().cpu()),
            },
            "evidence": table,
        }

        if show:
            display(HTML(
                f"<h3>Prediction: {labels[pred]}</h3>"
                f"<p><b>Non-Hate:</b> {result['probabilities']['Non-Hate']:.1%}"
                f" &nbsp; | &nbsp; <b>Hate:</b> {result['probabilities']['Hate']:.1%}</p>"
            ))
            display(table)
            display(HTML(
                "<small>GAT attention is a summary of attention allocated across graph edges. "
                "GAT/GCN contribution uses gradient × activation as an approximate contribution "
                "to the predicted-vs-other logit margin. Positive values support the prediction; "
                "negative values oppose it. These scores explain model behaviour and do not prove causality.</small>"
            ))
        return result
