import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# python -m streamlit run app.py

st.set_page_config(page_title="Vehicle Classifier", page_icon="🚗")
st.title("🚗 Vehicle Classifier")

# app.py and predict.py are inside src/
src_dir = Path(__file__).resolve().parent
project_dir = src_dir.parent
predict_file = src_dir / "predict.py"
saved_dir = project_dir / "results" / "saved"
test_dir = project_dir / "results" / "test"


# -----------------------------
# Model settings
# -----------------------------
st.sidebar.header("Model Settings")

# Load all checkpoints from ../results/saved
checkpoints = sorted(saved_dir.glob("*.pt")) if saved_dir.exists() else []

if checkpoints:
    checkpoint_options = [p.name for p in checkpoints]
    checkpoint_options.append("Custom path...")

    selected_checkpoint = st.sidebar.selectbox(
        "Checkpoint",
        checkpoint_options,
    )

    if selected_checkpoint == "Custom path...":
        checkpoint = st.sidebar.text_input(
            "Checkpoint path",
            placeholder="../results/saved/my_model.pt",
        )
    else:
        checkpoint = str(saved_dir / selected_checkpoint)
else:
    st.sidebar.warning("No checkpoint found in ../results/saved")
    checkpoint = st.sidebar.text_input(
        "Checkpoint path",
        value="../results/saved/best_baseline_small_cnn_0.0.pt",
    )

# These are the models currently supported by predict.py in this project.
model_options = [
    "small_cnn",
    "resnet18",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "depthwise_cnn",
]

model = st.sidebar.selectbox("Model", model_options)

transform = st.sidebar.selectbox(
    "Transform",
    ["baseline", "resnet"],
)

loss = st.sidebar.selectbox(
    "Loss",
    ["ce", "bce"],
)

# Needed only when the checkpoint was trained with non-default settings.
dropout = st.sidebar.number_input(
    "Dropout",
    min_value=0.0,
    max_value=0.9,
    value=0.0,
    step=0.1,
)

pooling = st.sidebar.selectbox(
    "Pooling",
    ["max", "avg"],
)

threshold = st.sidebar.slider(
    "Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
)


# -----------------------------
# Image upload
# -----------------------------
uploaded = st.file_uploader(
    "تصویر را انتخاب کنید",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    st.image(uploaded, caption="Input", width=350)

    if st.button("Predict", type="primary"):
        if not checkpoint:
            st.error("Checkpoint path را وارد یا انتخاب کنید.")
            st.stop()

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (src_dir / checkpoint_path).resolve()

        if not checkpoint_path.exists():
            st.error(f"Checkpoint پیدا نشد:\n{checkpoint_path}")
            st.stop()

        if not predict_file.exists():
            st.error(f"predict.py پیدا نشد:\n{predict_file}")
            st.stop()

        # Temporary image for predict.py
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded.name).suffix,
        ) as f:
            f.write(uploaded.getbuffer())
            image_path = Path(f.name)

        cmd = [
            sys.executable,
            str(predict_file),
            str(image_path),
            "--checkpoint",
            str(checkpoint_path),
            "--model",
            model,
            "--transform",
            transform,
            "--loss",
            loss,
            "--threshold",
            str(threshold),
            "--dropout",
            str(dropout),
            "--pooling",
            pooling,
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=src_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            output = result.stdout or result.stderr or "No output"

            if result.returncode == 0:
                st.success("Prediction completed")
                st.code(output)

                test_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = test_dir / f"prediction_{timestamp}.json"
                output_file.write_text(output, encoding="utf-8")

                st.caption(f"Saved: {output_file}")
            else:
                st.error("Prediction failed")
                st.code(output)

        finally:
            image_path.unlink(missing_ok=True)
