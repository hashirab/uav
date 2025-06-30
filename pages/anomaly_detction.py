import streamlit as st
import cv2
import torch
import os
import tempfile
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision.transforms import functional as F

# ─────────────────────────────────────────────────────────────
# Streamlit UI Setup
st.set_page_config(page_title="Anomaly Detection", layout="centered")
st.title("📹 Anomaly Detection with Faster R-CNN")

# ─────────────────────────────────────────────────────────────
# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 14  # 13 classes + background

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load("Anomaly/resnet50_final.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

model = load_model()

# ─────────────────────────────────────────────────────────────
# Utility
def draw_boxes(frame, outputs, threshold=0.5):
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label.item()}:{score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

# ─────────────────────────────────────────────────────────────
# Interface
video_file = st.file_uploader("Upload a video for anomaly detection", type=["mp4", "mov", "avi"])

if video_file:
    st.video(video_file)
    start_button = st.button("▶ Start Anomaly Detection")

    if start_button:
        stframe = st.empty()
        progress = st.progress(0)

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(video_file.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(tempfile.gettempdir(), "anomaly_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(image).to(device)

            with torch.no_grad():
                prediction = model([tensor])[0]

            annotated = draw_boxes(frame.copy(), prediction)
            out.write(annotated)

            resized = cv2.resize(annotated, (960, 540))
            stframe.image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), channels="RGB")

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()

        st.success("✅ Anomaly detection complete!")

        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Annotated Video",
                data=f,
                file_name="anomaly_output.mp4",
                mime="video/mp4"
            )
