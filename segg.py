import streamlit as st
import cv2
import torch
import tempfile
import os
import numpy as np
from torchvision import transforms as T
from PIL import Image
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────────────────────
# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_w, resize_h = 640, 384
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean, std)
])
color_map = np.random.RandomState(42).randint(0, 255, size=(23, 3), dtype=np.uint8)

def apply_mask(image, mask):
    mask_color = color_map[mask]
    return cv2.addWeighted(image, 0.6, mask_color, 0.4, 0)

# ─────────────────────────────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="Real-Time UNet Video Segmentation", layout="centered")
st.title("🎥 Real-Time UNet Segmentation")

# Model Selection
model_name = st.selectbox("Choose the segmentation model", ["MobileNet", "ResNet34"])
model_path_map = {
    "MobileNet": "unet_mobilenet_final_50.pt",
    "ResNet34": "unet_resnet34_final_50.pt"
}

# Load selected model
@st.cache_resource
def load_model(selected_model):
    model = smp.Unet(
        encoder_name="mobilenet_v2" if selected_model == "MobileNet" else "resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=23
    )
    model.load_state_dict(torch.load(model_path_map[selected_model], map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(model_name)

# Video Upload
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file:
    st.video(video_file)

    start_button = st.button("▶ Start Real-Time Segmentation")

    if start_button:
        stframe = st.empty()
        progress = st.progress(0)

        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        input_temp.write(video_file.read())

        cap = cv2.VideoCapture(input_temp.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(tempfile.gettempdir(), "segmented_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized = cv2.resize(frame, (resize_w, resize_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            mask_resized = cv2.resize(pred_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            overlay = apply_mask(frame, mask_resized)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            out.write(overlay)
            stframe.image(overlay_rgb, channels="RGB", use_column_width=True)

            frame_count += 1
            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        st.success("✅ Segmentation complete!")

        with open(output_path, 'rb') as f:
            st.download_button(
                label="📥 Download Segmented Video",
                data=f,
                file_name="segmented_output.mp4",
                mime="video/mp4"
            )
