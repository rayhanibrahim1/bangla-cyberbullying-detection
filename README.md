---
title: Bangla Cyberbullying Detection with Graph Evidence
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Bangla Cyberbullying Detection with Graph Evidence

Enter a Bangla sentence to see the model prediction, class probabilities, and word-level graph evidence from GAT and GCN.

- **Cyberbullying**: positive class (label 1)
- **Non-Cyberbullying**: negative class (label 0)

GAT attention summarizes attention allocation over graph edges. GAT/GCN contribution values are approximate gradient-times-activation scores. These describe model behaviour; they are not causal proof. Ensure this class mapping matches the dataset used to train the checkpoint.
