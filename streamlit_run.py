
import streamlit as st
from pathlib import Path
from datetime import datetime
import subprocess
import os
import argparse
import sys
sys.path.append('/content')

from scripts.inference import main
from omegaconf import OmegaConf

CONFIG_PATH = Path("configs/unet/stage2.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

st.set_page_config(page_title="LatentSync", layout="wide")
st.title("🎙️ LatentSync - Lip Sync Video Generator")

with st.sidebar:
    st.header("⚙️ Inference Settings")
    guidance_scale = st.slider("Guidance Scale", 1.0, 2.5, 1.5, 0.1)
    inference_steps = st.slider("Inference Steps", 10, 50, 20)
    seed = st.number_input("Random Seed", value=1247, step=1)

st.markdown("### 📤 Upload Files to Get Started")
video_file = st.file_uploader("📹 Upload a Video", type=["mp4", "mov"])
audio_file = st.file_uploader("🎵 Upload an Audio", type=["wav", "mp3"])

if st.button("🚀 Run LatentSync"):
    if video_file is None or audio_file is None:
        st.error("Please upload both video and audio files.")
        st.stop()

    st.info("Processing started...")

    input_dir = Path("temp_inputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    video_path = input_dir / video_file.name
    audio_path = input_dir / audio_file.name

    with open(video_path, "wb") as vf:
        vf.write(video_file.read())
    with open(audio_path, "wb") as af:
        af.write(audio_file.read())

    output_path = Path("temp_outputs") / f"synced_{datetime.now().strftime('%H%M%S')}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(CONFIG_PATH)
    config["run"].update({
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
    })

    args = argparse.Namespace(
        inference_ckpt_path=str(CHECKPOINT_PATH.resolve()),
        video_path=str(video_path.resolve()),
        audio_path=str(audio_path.resolve()),
        video_out_path=str(output_path.resolve()),
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    with st.spinner("🌀 Generating video... please wait"):
        try:
            main(config=config, args=args)
            st.success("✅ Video generated successfully!")
            st.video(str(output_path))

            if output_path.exists():
                with open(output_path, "rb") as f:
                    st.download_button("📥 Download Output Video", f, file_name=output_path.name)
        except Exception as e:
            st.error(f"❌ Inference failed: {e}")
