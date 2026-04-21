import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import timm

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MangoLeafVarietyBD",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  — one viewport desktop, mobile-app on ≤700 px
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800&display=swap');

/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --amber:    #f59e0b;
    --amber-d:  #b45309;
    --amber-l:  #fde68a;
    --green:    #16a34a;
    --green-d:  #14532d;
    --green-l:  #bbf7d0;
    --bg:       #0a0f0d;
    --bg2:      #0f1a13;
    --bg3:      #162019;
    --surface:  rgba(255,255,255,0.04);
    --border:   rgba(255,255,255,0.08);
    --txt:      #f0fdf4;
    --txt2:     #86efac;
    --txt3:     #6b7280;
    --danger:   #ef4444;
    --danger-bg:rgba(239,68,68,0.1);
}

/* ── FORCE LAYOUT TO ONE VIEWPORT ── */
html, body, [data-testid="stAppViewContainer"] {
    height: 100vh !important;
    overflow: hidden !important;
    background: var(--bg) !important;
}
[data-testid="stAppViewContainer"] > section.main {
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
}
.block-container {
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header[data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"] { display: none !important; }

/* ── full-height wrapper ── */
.app-shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    width: 100%;
    font-family: 'Outfit', sans-serif;
    background: var(--bg);
    color: var(--txt);
    overflow: hidden;
}

/* ── top navbar ── */
.navbar {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 56px;
    background: linear-gradient(90deg, var(--green-d) 0%, #0a3d1f 40%, #0a1a0d 100%);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    position: relative;
    overflow: hidden;
}
.navbar::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 100% at 75% 50%, rgba(245,158,11,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.nb-brand {
    display: flex; align-items: center; gap: 10px;
}
.nb-icon {
    font-size: 22px; line-height: 1;
}
.nb-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.3px;
}
.nb-title span { color: var(--amber) !important; }

.nb-pills {
    display: flex; gap: 8px;
}
.nb-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 11px; font-weight: 600;
    color: rgba(255,255,255,0.75) !important;
    letter-spacing: 0.3px;
}
.nb-pill.live {
    background: rgba(22,163,74,0.2);
    border-color: rgba(22,163,74,0.4);
    color: #4ade80 !important;
    display: flex; align-items: center; gap: 5px;
}
.nb-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 5px #4ade80;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.7)} }

/* ── body: two columns ── */
.app-body {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    min-height: 0;
    overflow: hidden;
}

/* ── panels ── */
.panel {
    display: flex;
    flex-direction: column;
    padding: 18px 22px;
    gap: 12px;
    overflow: hidden;
}
.panel-left {
    border-right: 1px solid var(--border);
    background: var(--bg2);
}
.panel-right {
    background: var(--bg);
}

.panel-heading {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--txt3) !important;
    flex-shrink: 0;
}

/* ── upload zone ── */
.upload-zone {
    flex: 1;
    border: 2px dashed rgba(245,158,11,0.35);
    border-radius: 16px;
    background: rgba(245,158,11,0.03);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 10px; cursor: pointer;
    transition: border-color .2s, background .2s;
    min-height: 0;
    position: relative; overflow: hidden;
}
.upload-zone:hover {
    border-color: var(--amber);
    background: rgba(245,158,11,0.06);
}
.uz-icon {
    width: 48px; height: 48px;
    background: rgba(245,158,11,0.12);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.uz-text { font-size: 13px; color: rgba(255,255,255,0.55) !important; text-align: center; line-height: 1.5; }
.uz-hint { font-size: 11px; color: var(--txt3) !important; }

/* ── info strip below upload ── */
.info-strip {
    flex-shrink: 0;
    display: flex; gap: 8px; flex-wrap: wrap;
}
.info-chip {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 99px; padding: 4px 11px;
    font-size: 11px; font-weight: 500;
    color: rgba(255,255,255,0.55) !important;
}
.ic-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--amber); }

/* ── analyse button ── */
.analyse-btn-wrap { flex-shrink: 0; }

/* ── right panel states ── */
.idle-state {
    flex: 1;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 12px;
    border: 1px dashed var(--border);
    border-radius: 20px;
    min-height: 0;
}
.idle-icon { font-size: 40px; opacity: 0.3; }
.idle-text { font-size: 13px; color: var(--txt3) !important; text-align: center; line-height: 1.7; }

.result-card {
    flex: 1;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px 24px;
    display: flex; flex-direction: column;
    gap: 14px; min-height: 0; overflow: hidden;
}

.result-tag {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4ade80 !important; padding: 4px 0;
}
.result-tag-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #4ade80;
    box-shadow: 0 0 6px #4ade80;
}

.variety-name {
    font-family: 'Playfair Display', serif;
    font-size: clamp(28px, 3.5vw, 44px);
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.1;
    letter-spacing: -0.5px;
}

.conf-row {
    display: flex; align-items: baseline; gap: 8px;
}
.conf-val {
    font-size: 28px; font-weight: 700;
    color: var(--amber) !important;
}
.conf-lbl {
    font-size: 13px;
    color: var(--txt3) !important;
}

.conf-bar-wrap {
    height: 6px; border-radius: 3px;
    background: rgba(255,255,255,0.07); overflow: hidden;
}
.conf-bar {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, var(--green), var(--amber));
    transition: width .6s ease;
}

.model-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    flex-shrink: 0;
}
.model-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 10px;
    display: flex; flex-direction: column; gap: 2px;
}
.mc-name { font-size: 10px; font-weight: 600; color: var(--txt3) !important; text-transform: uppercase; letter-spacing: .5px; }
.mc-val  { font-size: 14px; font-weight: 700; color: var(--txt) !important; }

.error-card {
    flex: 1;
    background: var(--danger-bg);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px; padding: 24px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 10px; text-align: center; min-height: 0;
}
.err-icon { font-size: 36px; }
.err-title { font-size: 16px; font-weight: 700; color: var(--danger) !important; }
.err-body  { font-size: 13px; color: rgba(255,255,255,0.6) !important; line-height: 1.6; }

/* ── footer strip ── */
.app-footer {
    flex-shrink: 0;
    background: var(--green-d);
    padding: 6px 28px;
    display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 6px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.af-text { font-size: 11px; color: rgba(255,255,255,0.4) !important; }
.af-text strong { color: var(--amber) !important; }

/* ── override Streamlit widget chrome ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: 2px dashed rgba(245,158,11,0.35) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--amber) !important;
    background: rgba(245,158,11,0.04) !important;
}
[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
}
[data-testid="stFileUploader"] button {
    background: var(--amber) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 9px !important;
}
[data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] p {
    color: rgba(255,255,255,0.65) !important;
}
[data-testid="stImage"] img {
    border-radius: 12px !important;
    width: 100% !important;
    max-height: 200px !important;
    object-fit: cover !important;
}

/* Streamlit button */
.stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important; font-weight: 700 !important;
    width: 100% !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 16px rgba(22,163,74,0.3) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(22,163,74,0.42) !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #16a34a, #f59e0b) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--amber) !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--amber) !important; }

/* ════ MOBILE  (≤700px) ════════════════════════════════════════ */
@media (max-width: 700px) {
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section.main,
    .block-container {
        height: auto !important;
        overflow: auto !important;
    }
    .app-shell { height: auto !important; overflow: auto !important; }
    .navbar { height: auto; padding: 12px 16px; flex-wrap: wrap; gap: 8px; }
    .nb-pills { flex-wrap: wrap; }
    .app-body {
        grid-template-columns: 1fr !important;
        overflow: auto !important;
        height: auto !important;
    }
    .panel { padding: 14px 14px; }
    .panel-left { border-right: none; border-bottom: 1px solid var(--border); }
    .upload-zone { min-height: 150px; }
    .variety-name { font-size: 30px; }
    .model-grid { grid-template-columns: 1fr 1fr; }
    .app-footer { flex-direction: column; gap: 3px; padding: 10px 16px; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_engine():
    bundle_path = "Hybrid-MangoLeaf_bundle.pth"
    if not os.path.exists(bundle_path):
        return None, None
    bundle = torch.load(bundle_path, map_location="cpu")
    num_classes = len(bundle["classes"])
    loaded = {}
    for name in bundle["model_names"]:
        if name == "EfficientNetB0":
            m = models.efficientnet_b0(weights=None)
            m.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(m.classifier[1].in_features, num_classes)
            )
        elif name == "MobileNetV2":
            m = models.mobilenet_v2(weights=None)
            m.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(m.classifier[1].in_features, num_classes)
            )
        elif name == "DeiT-Tiny":
            m = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        elif name == "Swin-Tiny":
            m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
        else:
            continue
        if name in bundle["states"]:
            m.load_state_dict(bundle["states"][name])
        m.eval()
        loaded[name] = m
    return loaded, bundle

models_engine, bundle_meta = load_engine()
engine_ok = models_engine is not None

THRESHOLD = 65.0
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(img: Image.Image):
    tensor = TRANSFORM(img).unsqueeze(0)
    per_model = {}
    all_p = []
    with torch.no_grad():
        for name, m in models_engine.items():
            p = F.softmax(m(tensor), dim=1)
            w = bundle_meta["weights"].get(name, 0.0)
            per_model[name] = float(torch.max(p).item()) * 100
            all_p.append(p * w)
        final_p = torch.stack(all_p).sum(dim=0)
        conf, idx = torch.max(final_p, 1)
    variety = bundle_meta["classes"][idx.item()]
    score   = float(conf.item()) * 100
    return variety, score, per_model


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    live_status = "ONLINE" if engine_ok else "OFFLINE"
    live_class  = "live" if engine_ok else ""

    # ── Navbar ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="app-shell">
    <nav class="navbar">
        <div class="nb-brand">
            <span class="nb-icon">🥭</span>
            <span class="nb-title">MangoLeaf<span>VarietyBD</span></span>
        </div>
        <div class="nb-pills">
            <span class="nb-pill">Hybrid Ensemble ×4</span>
            <span class="nb-pill">MangoLeafVarietyBD Dataset</span>
            <span class="nb-pill {live_class}">
                {'<span class="nb-dot"></span>' if engine_ok else ''}
                {live_status}
            </span>
        </div>
    </nav>
    """, unsafe_allow_html=True)

    # ── Body ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="app-body">', unsafe_allow_html=True)

    # Left panel
    st.markdown('<div class="panel panel-left">', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Input sample</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload mango leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div class="uz-icon">🍃</div>
            <div class="uz-text">Drop mango leaf image here<br>or use the uploader above</div>
            <div class="uz-hint">JPG · JPEG · PNG</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-strip">
        <span class="info-chip"><span class="ic-dot"></span>224×224 inference</span>
        <span class="info-chip"><span class="ic-dot"></span>4-model ensemble</span>
        <span class="info-chip"><span class="ic-dot"></span>65% threshold</span>
    </div>
    """, unsafe_allow_html=True)

    analyse_clicked = False
    if uploaded:
        analyse_clicked = st.button("🔬  Analyse Leaf", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close panel-left

    # Right panel
    st.markdown('<div class="panel panel-right">', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">Detection result</div>', unsafe_allow_html=True)

    if analyse_clicked and uploaded:
        if not engine_ok:
            st.markdown("""
            <div class="error-card">
                <div class="err-icon">⚠️</div>
                <div class="err-title">Engine Offline</div>
                <div class="err-body">
                    Hybrid-MangoLeaf_bundle.pth not found.<br>
                    Place the bundle file in the same directory as this script.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Running ensemble inference…"):
                variety, score, per_model = run_inference(img)

            st.session_state["result"] = (variety, score, per_model)

    if "result" in st.session_state and uploaded:
        variety, score, per_model = st.session_state["result"]

        if score < THRESHOLD:
            st.markdown(f"""
            <div class="error-card">
                <div class="err-icon">❌</div>
                <div class="err-title">Invalid Input Detected</div>
                <div class="err-body">
                    Confidence {score:.1f}% is below the {THRESHOLD:.0f}% threshold.<br>
                    This may not be a recognisable mango leaf.<br>
                    Please upload a clear image on a plain background.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Build model chips HTML
            chips_html = ""
            for mname, mconf in per_model.items():
                chips_html += f"""
                <div class="model-chip">
                    <span class="mc-name">{mname}</span>
                    <span class="mc-val">{mconf:.1f}%</span>
                </div>"""

            st.markdown(f"""
            <div class="result-card">
                <div class="result-tag">
                    <span class="result-tag-dot"></span>
                    Variety Identified
                </div>
                <div class="variety-name">{variety}</div>
                <div class="conf-row">
                    <span class="conf-val">{score:.1f}%</span>
                    <span class="conf-lbl">ensemble confidence</span>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar" style="width:{min(score,100):.1f}%"></div>
                </div>
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                            letter-spacing:.9px;color:rgba(255,255,255,0.35);margin-top:4px;">
                    Per-model scores
                </div>
                <div class="model-grid">{chips_html}</div>
            </div>
            """, unsafe_allow_html=True)

            if score > 85:
                st.balloons()
    else:
        st.markdown("""
        <div class="idle-state">
            <div class="idle-icon">🍃</div>
            <div class="idle-text">Neural engine idle<br>Upload a sample and click Analyse</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close panel-right
    st.markdown("</div>", unsafe_allow_html=True)  # close app-body

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
        <span class="af-text">
            Developed by <strong>Habibur Rahman Sajal</strong>
            &nbsp;·&nbsp; MangoLeafVarietyBD Dataset
        </span>
        <span class="af-text">
            EfficientNetB0 · MobileNetV2 · DeiT-Tiny · Swin-Tiny
        </span>
    </div>
    </div>
    """, unsafe_allow_html=True)  # last </div> closes app-shell


if __name__ == "__main__":
    main()
