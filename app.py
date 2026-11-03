from pathlib import Path
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer

from model import HybridNet
from explainability import Explainer

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ROOT / "model.pkl"
TOKENIZER_PATH = ROOT / "tokenizer"

COLUMNS = [
    "Word",
    "GAT attention (%)",
    "GAT contribution*",
    "GCN contribution*",
]
EMPTY_TABLE = pd.DataFrame(columns=COLUMNS)

LABELS = {0: "Non-Cyberbullying", 1: "Cyberbullying"}

st.set_page_config(
    page_title="Bangla Cyberbullying Detection with Graph Evidence",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1050px !important; margin: auto !important;}
    #main-title {text-align: center; margin-bottom: 0.2rem;}
    #subtitle {text-align: center; color: gray; margin-bottom: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Model ebong Tokenizer load hocche...")
def load_explainer():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "model.pkl is missing. Upload the trained model.pkl exported from Colab."
        )
    if not TOKENIZER_PATH.is_dir():
        raise FileNotFoundError(
            "tokenizer/ is missing. Upload the tokenizer folder exported from Colab."
        )
    if not (ROOT / "backbone_config").is_dir():
        raise FileNotFoundError(
            "backbone_config/ is missing. Upload the folder exported from Colab."
        )

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(
            "Unsupported model.pkl format: expected keys 'config' and 'state_dict'. "
            "Export the package from the trained Colab runtime."
        )

    model = HybridNet(checkpoint["config"])
    state = dict(checkpoint["state_dict"])

    for old_prefix, new_prefix in (("norm1.", "n1."), ("norm2.", "n2.")):
        for key in list(state.keys()):
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                if new_key in state:
                    raise RuntimeError(
                        f"Checkpoint contains both {key!r} and {new_key!r}; "
                        "please check which model export was uploaded."
                    )
                state[new_key] = state.pop(key)

    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH), use_fast=True, local_files_only=True
    )
    return Explainer(model, tokenizer, checkpoint["config"], device=DEVICE)


def predict_and_explain(text, explainer):
    """Return UI labels, probabilities, and word-level GAT/GCN evidence."""
    if not text or not text.strip():
        return (
            "### একটি বাংলা বাক্য লিখুন\nবিশ্লেষণ শুরু করতে বাক্য লিখে Detect and Show Evidence চাপুন।",
            {},
            EMPTY_TABLE,
        )

    try:
        result = explainer.explain(text, show=False)
        raw_probs = result["probabilities"]
        raw_pred = result["prediction"]

        p_non = float(
            raw_probs.get(
                "Non-Cyberbullying",
                raw_probs.get("Non-Hate Speech", raw_probs.get("Non-Hate", 0.0)),
            )
        )
        p_cb = float(
            raw_probs.get(
                "Cyberbullying",
                raw_probs.get("Hate Speech", raw_probs.get("Hate", 0.0)),
            )
        )
        probs = {
            "Non-Cyberbullying": p_non,
            "Cyberbullying": p_cb,
        }
        pred_map = {
            "Non-Cyberbullying": "Non-Cyberbullying",
            "Cyberbullying": "Cyberbullying",
            "Non-Hate": "Non-Cyberbullying",
            "Hate": "Cyberbullying",
            "Non-Hate Speech": "Non-Cyberbullying",
            "Hate Speech": "Cyberbullying",
        }
        if isinstance(raw_pred, int):
            pred = LABELS.get(raw_pred, LABELS[1] if p_cb >= p_non else LABELS[0])
        else:
            pred = pred_map.get(raw_pred, LABELS[1] if p_cb >= p_non else LABELS[0])

        confidence = max(probs.values())
        table = result["evidence"]

        summary = (
            f"#### মডেলের ফলাফল: **{pred}**\n\n"
            f"**Confidence:** {confidence:.2%}\n\n"
            "*এটি মডেলের prediction—মানুষের চূড়ান্ত বিচার নয়।*"
        )
        return summary, probs, table
    except Exception as exc:
        print(f"Prediction/explanation error: {type(exc).__name__}: {exc}")
        return (
            f"### বিশ্লেষণ করা যায়নি\n`{type(exc).__name__}: {exc}`",
            {},
            EMPTY_TABLE,
        )


st.markdown(
    "<h1 id='main-title'>Bangla Cyberbullying Detection with Graph Evidence</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p id='subtitle'>বাংলা বাক্য লিখুন—মডেলের cyberbullying prediction, probability এবং GAT/GCN word-level graph evidence দেখুন।</p>",
    unsafe_allow_html=True,
)

try:
    explainer = load_explainer()
except Exception as e:
    st.error(f"Model load korte somoshya hoyeche: {e}")
    st.stop()

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

col1, col2 = st.columns([3, 2])

with col1:
    text_box = st.text_area(
        label="বাংলা বাক্য",
        value=st.session_state.input_text,
        placeholder="এখানে বাংলা বাক্য লিখুন...",
        height=130,
    )
    submit = st.button("Detect and Show Evidence", type="primary", use_container_width=True)

    st.markdown("**Examples (উদাহরণ হিসেবে ক্লিক করুন):**")
    ex_col1, ex_col2 = st.columns(2)
    if ex_col1.button("তুমি সবসময় অন্যদের অপমান করো।", use_container_width=True):
        st.session_state.input_text = "তুমি সবসময় অন্যদের অপমান করো।"
        st.rerun()
    if ex_col2.button("আজকের আবহাওয়া সুন্দর।", use_container_width=True):
        st.session_state.input_text = "আজকের আবহাওয়া সুন্দর।"
        st.rerun()

if submit or text_box:
    summary, probs, table = predict_and_explain(text_box, explainer)
else:
    summary, probs, table = predict_and_explain("", explainer)

with col2:
    st.markdown(summary)
    if probs:
        st.markdown("**Cyberbullying class probabilities**")
        for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{label}: **{prob:.2%}**")
            st.progress(min(max(prob, 0.0), 1.0))

st.subheader("Word-level evidence (GAT / GCN)")
st.dataframe(table, use_container_width=True, hide_index=True)

st.markdown(
    "### Evidence guide\n"
    "- **GAT attention (%)**: graph-এর word-node-গুলোর attention allocation-এর সারাংশ।\n"
    "- **GAT contribution*** এবং **GCN contribution***: gradient × activation-ভিত্তিক আনুমানিক contribution score।\n\n"
    "এগুলো মডেলের আচরণ বোঝাতে সাহায্য করে; কোনো শব্দ prediction-এর নিশ্চিত কারণ—এটি প্রমাণ করে না।"
)
st.markdown(
    "**Class mapping:** এখানে class 0 = Non-Cyberbullying এবং class 1 = Cyberbullying ধরে দেখানো হয়। তোমার training dataset-এর label mapping-এর সঙ্গে এটি মিলিয়ে নিও।"
)
st.markdown(
    "Model architecture: BanglaBERT + Graph Attention Network (GAT) + Graph Convolutional Network (GCN)."
)
