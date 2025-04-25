import streamlit as st
from glob import glob
import os
import numpy as np
import tifffile
from penguin_preprocess import filters as IPrep
from penguin_preprocess import parser as IP
from streamlit_image_comparison import image_comparison
from streamlit_image_zoom import image_zoom
from streamlit_cropper import st_cropper  # pip install streamlit-cropper
from PIL import Image
from pathlib import Path

# Get the directory where app.py is located
app_dir = Path(__file__).resolve().parent

# Define default folders relative to app_dir
default_input_folder = os.path.join(app_dir, 'demodata-raw')
default_output_folder = os.path.join(app_dir, 'demodata-preprocessed')
os.mkdirs(default_output_folder, exist_ok=True)
st.set_page_config(layout="wide")
st.title("🐧 PENGUIN Preprocessing Streamlit")

# ─── Layout ──────────────────────────────────────────────────────────────────
col_sidebar, col_center, col_tracker = st.columns([1, 3.2, 0.6])

# ─── Sidebar: data + parameters ─────────────────────────────────────────────
with col_sidebar:
    input_folder = st.text_input("Input folder", default_input_folder)
    output_folder = st.text_input("Output folder", default_output_folder)

all_files = glob(os.path.join(input_folder, "*", "*.ome.tiff"))
col_sidebar.text(f"🔍 {len(all_files)} files found")
if not all_files:
    col_center.warning("⚠️ No .ome.tiff files found.")
    st.stop()

channels = sorted({os.path.splitext(os.path.basename(f))[0] for f in all_files})
fovs     = sorted({os.path.basename(os.path.dirname(f))      for f in all_files})

if "channel_index" not in st.session_state:
    st.session_state.channel_index = 0
if "constants_tracker" not in st.session_state:
    st.session_state.constants_tracker = {}

with col_sidebar:
    selected_channel = st.selectbox("Channel", channels, index=st.session_state.channel_index)
    threshold        = st.slider("Background Threshold", 0.0, 1.0, 0.1, 0.05)
    percentile       = st.slider("Percentile filter", 0, 100, 50)
    selected_fov     = st.selectbox("Field of View (FOV)", fovs)

# ─── Load & preprocess a preview image ───────────────────────────────────────
try:
    preview_file = next(f for f in all_files if selected_fov in f and selected_channel in f)
except StopIteration:
    col_center.error("⚠️ No matching file.")
    st.stop()

img_orig = IP.parse_image(preview_file)
if img_orig is None or img_orig.ndim < 2:
    col_center.error("❌ Failed to parse preview.")
    st.stop()
if img_orig.ndim == 2:
    img_orig = np.expand_dims(img_orig, -1)

img_proc = IPrep.normalize_channel_cv2_minmax(IPrep.remove_outliers(img_orig))
img_proc = IPrep.out_ratio2(img_proc, th=threshold)
img_proc = IPrep.percentile_filter(img_proc, percentile=percentile)

def first_channel_preview(img):
    return (np.squeeze(img[..., 0]) * 255).astype(np.uint8)

orig_pil = Image.fromarray(first_channel_preview(img_orig))
proc_pil = Image.fromarray(first_channel_preview(img_proc))

# ─── Central panel: Slider vs Interactive Zoom ──────────────────────────────
with col_center:
    st.markdown("### Image Preview")

    # 1) Zoom level
    zoom_factor = st.slider("🔍 Zoom factor", min_value=1, max_value=20, value=1, step=1)

    # 2) Convert to PIL for slider comparison
    orig_pil = Image.fromarray(first_channel_preview(img_orig))
    proc_pil = Image.fromarray(first_channel_preview(img_proc))

    # 3) Apply zoom
    if zoom_factor > 1:
        w, h = orig_pil.size
        center_x, center_y = w // 2, h // 2
        zoomed_w, zoomed_h = w // zoom_factor, h // zoom_factor
        x1 = max(0, center_x - zoomed_w // 2)
        y1 = max(0, center_y - zoomed_h // 2)
        x2 = min(w, center_x + zoomed_w // 2)
        y2 = min(h, center_y + zoomed_h // 2)

        orig_crop = orig_pil.crop((x1, y1, x2, y2))
        proc_crop = proc_pil.crop((x1, y1, x2, y2))

        display_w, display_h = w, h  # maintain full container size
        orig_display = orig_crop.resize((display_w, display_h), resample=Image.NEAREST)
        proc_display = proc_crop.resize((display_w, display_h), resample=Image.NEAREST)
    else:
        orig_display = orig_pil
        proc_display = proc_pil

    # 4) Show side-by-side slider
    image_comparison(
        img1=orig_display,
        img2=proc_display,
        label1="Original",
        label2="Processed",
        width=950
    )


# ─── Sidebar: Save constants ─────────────────────────────────────────────────
with col_sidebar:
    if st.button("💾 Save constants for this marker"):
        st.session_state.constants_tracker[selected_channel] = {
            "threshold": threshold,
            "percentile": percentile
        }
        st.success(f"Saved constants for {selected_channel}")
        st.session_state.channel_index = (st.session_state.channel_index + 1) % len(channels)
        st.rerun()

# ─── Tracker ────────────────────────────────────────────────────────────────
with col_tracker:
    st.markdown("<h5>📋 Tracker</h5>", unsafe_allow_html=True)
    for m, vals in st.session_state.constants_tracker.items():
        st.markdown(f"<small><b>{m}</b>: th={vals['threshold']}, p={vals['percentile']}</small>", unsafe_allow_html=True)
    missing = [m for m in channels if m not in st.session_state.constants_tracker]
    if missing:
        st.warning(f"⚠️ Missing: {', '.join(missing)}")

# ─── Run Batch ──────────────────────────────────────────────────────────────
if col_center.button("Run full preprocessing"):
    tracker = st.session_state.constants_tracker
    missing = [c for c in channels if c not in tracker]
    if missing:
        col_center.error(f"⚠️ Define constants for: {', '.join(missing)}")
        st.stop()

    total = sum(1 for f in all_files if any(ch in f for ch in tracker))
    prog = col_center.progress(0)
    cnt  = 0
    for m, const in tracker.items():
        for f in [x for x in all_files if m in x]:
            img = IP.parse_image(f)
            if img is None or img.ndim < 2:
                continue
            if img.ndim == 2:
                img = np.expand_dims(img, -1)

            img = IPrep.normalize_channel_cv2_minmax(IPrep.remove_outliers(img))
            img = IPrep.out_ratio2(img, th=const["threshold"])
            img = IPrep.percentile_filter(img, percentile=const["percentile"])

            dest = f.replace(input_folder, output_folder)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            tifffile.imwrite(dest, img)

            cnt += 1
            prog.progress(cnt / total)

    col_center.success("✅ Full preprocessing completed!")
