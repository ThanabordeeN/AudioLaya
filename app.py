from __future__ import annotations

import hashlib
import mimetypes
import random
import tempfile
from pathlib import Path

import streamlit as st

from src.data.splits import read_manifests
from src.inference import classify_audio_file, load_classifier

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "poc.yaml"
CHECKPOINT = ROOT / "runs" / "projector_head" / "best.pt"
MANIFESTS = [ROOT / "data" / "ftc.jsonl", ROOT / "data" / "harper.jsonl"]


@st.cache_resource(show_spinner=False)
def get_classifier():
    return load_classifier(CHECKPOINT, CONFIG)


@st.cache_data(show_spinner=False)
def get_heldout_samples():
    rows = read_manifests(MANIFESTS)
    samples = [row for row in rows if row["split"] == "test"]
    random.Random(42).shuffle(samples)
    return samples


st.set_page_config(page_title="Audio-Laya", page_icon="📞", layout="centered")
st.title("Audio-Laya")
st.write("Classify a call directly from its audio—no transcript is generated.")
st.warning(
    "Research PoC only. Spam and legitimate examples come from different datasets, "
    "so test scores may reflect dataset differences rather than call content."
)

source = st.radio("Audio source", ["Unseen test sample", "Upload audio"], horizontal=True)
uploaded = None
audio_path = None
audio_bytes = None
audio_key = None
ground_truth = None
ground_truth_source = None

if source == "Upload audio":
    uploaded = st.file_uploader(
        "Choose a call recording",
        type=["wav", "flac", "ogg", "aiff", "aif"],
        help="Mono or stereo; audio is downmixed and resampled to 16 kHz, then processed in chunks up to 30 seconds.",
    )
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        audio_key = "upload:" + hashlib.sha256(audio_bytes).hexdigest()
        mime_type = mimetypes.guess_type(uploaded.name)[0] or "audio/wav"
        st.audio(audio_bytes, format=mime_type)
else:
    try:
        samples = get_heldout_samples()
    except (OSError, ValueError) as exc:
        samples = []
        st.error(f"Could not read the held-out manifests: {exc}")
    if samples:
        st.caption(f"{len(samples)} original test calls held out from training; the answer is hidden until after classification.")
        index = st.selectbox(
            "Choose a held-out call",
            range(len(samples)),
            format_func=lambda i: f"Test call {i + 1:03d}",
        )
        row = samples[index]
        audio_path = Path(row["audio_path"])
        audio_key = f"test:{row['source']}:{row['call_id']}"
        ground_truth = "spam" if int(row["label"]) == 1 else "legitimate"
        ground_truth_source = row["source"]
        if audio_path.is_file():
            audio_bytes = audio_path.read_bytes()
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.error(f"Audio file missing: {audio_path}")
            audio_key = None
    else:
        st.error("No held-out calls found. Run the FTC and Harper manifest preparation commands first.")

if st.session_state.get("active_audio_key") != audio_key:
    st.session_state["active_audio_key"] = audio_key
    st.session_state.pop("result", None)
    st.session_state.pop("result_key", None)

if not CHECKPOINT.is_file():
    st.error(f"Trained checkpoint not found: {CHECKPOINT}")

if st.button("Classify audio", type="primary", disabled=audio_key is None or not CHECKPOINT.is_file()):
    try:
        with st.spinner("Loading the model and classifying the recording…"):
            classifier = get_classifier()
            if audio_path is not None:
                result = classify_audio_file(classifier, audio_path)
            else:
                suffix = Path(uploaded.name).suffix.lower()
                with tempfile.NamedTemporaryFile(suffix=suffix) as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio.flush()
                    result = classify_audio_file(classifier, temp_audio.name)
        st.session_state["result"] = result
        st.session_state["result_key"] = audio_key
        st.session_state["ground_truth"] = ground_truth
        st.session_state["ground_truth_source"] = ground_truth_source
    except Exception as exc:
        st.error(f"Could not classify this file: {type(exc).__name__}: {exc}")

result = st.session_state.get("result")
if result and st.session_state.get("result_key") == audio_key:
    label = "Spam / Robocall" if result["label"] == "spam" else "Legitimate call"
    st.subheader(f"Decision: {label}")
    st.caption("Estimated class scores (not calibrated confidence)")
    left, right = st.columns(2)
    left.metric("Legitimate", f"{result['probabilities']['legitimate']:.1%}")
    right.metric("Spam / Robocall", f"{result['probabilities']['spam']:.1%}")
    st.caption(
        f"Processed {result['chunks']} audio chunk(s) · "
        f"{result['inference_latency_ms']:.0f} ms · transcript generated: no"
    )
    if st.session_state.get("ground_truth") is not None:
        truth = st.session_state["ground_truth"]
        source_name = "FTC" if st.session_state["ground_truth_source"] == "ftc" else "HarperValleyBank"
        verdict = "correct" if result["label"] == truth else "incorrect"
        st.info(f"Ground truth: {truth} ({source_name}) — prediction {verdict}.")
