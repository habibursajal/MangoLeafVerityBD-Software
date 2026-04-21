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
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --c-bg:       #060a07;
    --c-bg2:      #0b130d;
    --c-bg3:      #111c13;
    --c-bg4:      #162019;
    --c-green:    #22c55e;
    --c-green-d:  #15803d;
    --c-green-dd: #14532d;
    --c-amber:    #f59e0b;
    --c-amber-l:  #fcd34d;
    --c-border:   rgba(255,255,255,0.07);
    --c-border2:  rgba(255,255,255,0.12);
    --c-txt:      #f0fdf4;
    --c-txt2:     #a3e6b8;
    --c-txt3:     #4b7a5c;
    --c-danger:   #f87171;
    --c-danger-bg:rgba(248,113,113,0.08);
    --font-ui:    'DM Sans', sans-serif;
    --font-disp:  'Syne', sans-serif;
}

/* ── STREAMLIT SHELL OVERRIDE ── */
html, body { background: var(--c-bg) !important; }

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section.main {
    background: var(--c-bg) !important;
    padding: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* hide chrome */
#MainMenu, footer,
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebar"] { display: none !important; }

/* ══════════════════════════════════
   DESKTOP LAYOUT  (> 768 px)
   Full viewport in 3 rows:
   navbar | body (left+right) | footer
═══════════════════════════════════ */
.shell {
    font-family: var(--font-ui);
    color: var(--c-txt);
    background: var(--c-bg);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}

/* ── NAV ── */
.nav {
    flex-shrink: 0;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    background: linear-gradient(90deg, #0b2010 0%, #091a0c 60%, var(--c-bg) 100%);
    border-bottom: 1px solid var(--c-border);
    position: relative;
    z-index: 10;
}
.nav::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--c-green-d), transparent 60%);
}
.nav-brand {
    display: flex; align-items: center; gap: 9px;
}
.nav-brand-icon { font-size: 20px; }
.nav-brand-name {
    font-family: var(--font-disp);
    font-size: 16px;
    font-weight: 800;
    letter-spacing: -0.2px;
    color: #fff;
}
.nav-brand-name em {
    font-style: normal;
    color: var(--c-amber);
}
.nav-tags {
    display: flex; gap: 6px; align-items: center;
}
.nav-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 3px 10px;
    border-radius: 99px;
    border: 1px solid var(--c-border2);
    color: rgba(255,255,255,0.55);
    background: rgba(255,255,255,0.04);
}
.nav-tag.status-on {
    border-color: rgba(34,197,94,0.4);
    background: rgba(34,197,94,0.1);
    color: var(--c-green);
    display: flex; align-items: center; gap: 5px;
}
.nav-tag.status-off {
    border-color: rgba(248,113,113,0.4);
    background: rgba(248,113,113,0.1);
    color: var(--c-danger);
}
.blink {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--c-green);
    box-shadow: 0 0 6px var(--c-green);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── BODY ── */
.body {
    flex: 1;
    display: grid;
    grid-template-columns: 420px 1fr;
    min-height: 0;
    overflow: hidden;
}

/* ── PANELS ── */
.panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 16px 20px;
    gap: 10px;
}
.panel-left {
    background: var(--c-bg2);
    border-right: 1px solid var(--c-border);
}
.panel-right {
    background: var(--c-bg);
    padding: 16px 22px;
}

.panel-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--c-txt3);
    flex-shrink: 0;
    display: flex; align-items: center; gap: 7px;
}
.panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--c-border);
}

/* ── IMAGE AREA ── */
.img-wrap {
    flex: 1;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--c-border);
    background: var(--c-bg3);
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
    position: relative;
}
.img-placeholder {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 10px; height: 100%;
}
.img-placeholder-icon {
    width: 52px; height: 52px;
    border-radius: 50%;
    background: rgba(34,197,94,0.08);
    border: 1px dashed rgba(34,197,94,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
}
.img-placeholder-text {
    font-size: 12px;
    color: var(--c-txt3);
    text-align: center;
    line-height: 1.6;
}

/* ── UPLOADER OVERRIDE ── */
[data-testid="stFileUploader"] {
    flex-shrink: 0 !important;
}
[data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stFileUploader"] section {
    background: var(--c-bg3) !important;
    border: 1px dashed rgba(245,158,11,0.3) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    min-height: unset !important;
    transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--c-amber) !important;
    background: rgba(245,158,11,0.04) !important;
}
[data-testid="stFileUploader"] section > div {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    flex-direction: row !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p {
    color: rgba(255,255,255,0.55) !important;
    font-family: var(--font-ui) !important;
    font-size: 12px !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(245,158,11,0.15) !important;
    color: var(--c-amber) !important;
    border: 1px solid rgba(245,158,11,0.35) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    padding: 5px 12px !important;
    white-space: nowrap !important;
}

/* ── CHIPS ROW ── */
.chips-row {
    display: flex; gap: 6px; flex-wrap: wrap;
    flex-shrink: 0;
}
.chip {
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.3px;
    padding: 3px 10px; border-radius: 99px;
    border: 1px solid var(--c-border);
    color: rgba(255,255,255,0.4);
    background: rgba(255,255,255,0.03);
    display: flex; align-items: center; gap: 5px;
}
.chip-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--c-amber); }

/* ── ANALYSE BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, var(--c-green-d) 0%, #166534 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 0 !important;
    font-family: var(--font-disp) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(21,128,61,0.25) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(21,128,61,0.38) !important;
}

/* ── RIGHT PANEL STATES ── */
.idle-box {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    border: 1px dashed var(--c-border);
    border-radius: 18px;
    min-height: 0;
}
.idle-glyph { font-size: 36px; opacity: 0.2; }
.idle-msg {
    font-size: 12px;
    color: var(--c-txt3);
    text-align: center;
    line-height: 1.8;
}

/* ── RESULT CARD ── */
.rcard {
    flex: 1;
    background: var(--c-bg3);
    border: 1px solid var(--c-border);
    border-radius: 18px;
    padding: 22px 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
    overflow: hidden;
    position: relative;
}
.rcard::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--c-green-d), var(--c-amber), transparent 80%);
    border-radius: 18px 18px 0 0;
}

.r-tag {
    display: flex; align-items: center; gap: 6px;
    font-size: 9px; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: var(--c-green);
}
.r-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--c-green);
    box-shadow: 0 0 8px var(--c-green);
}

.r-variety {
    font-family: var(--font-disp);
    font-size: clamp(26px, 3.2vw, 42px);
    font-weight: 800;
    color: #fff;
    line-height: 1.05;
    letter-spacing: -0.5px;
}

.r-conf-row {
    display: flex; align-items: baseline; gap: 8px;
}
.r-conf-num {
    font-family: var(--font-disp);
    font-size: 32px; font-weight: 800;
    color: var(--c-amber);
    line-height: 1;
}
.r-conf-lbl {
    font-size: 12px;
    color: var(--c-txt3);
}

.r-bar-track {
    height: 5px; border-radius: 3px;
    background: rgba(255,255,255,0.06);
    overflow: hidden; flex-shrink: 0;
}
.r-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--c-green-d), var(--c-amber));
    transition: width .7s cubic-bezier(.4,0,.2,1);
}

.r-divider {
    height: 1px;
    background: var(--c-border);
    flex-shrink: 0;
}

.r-models-label {
    font-size: 9px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase;
    color: var(--c-txt3); flex-shrink: 0;
}

.r-model-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    flex-shrink: 0;
}
.r-mchip {
    background: var(--c-bg4);
    border: 1px solid var(--c-border);
    border-radius: 10px;
    padding: 9px 10px;
    display: flex; flex-direction: column; gap: 3px;
}
.r-mchip-name {
    font-size: 9px; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
    color: var(--c-txt3);
}
.r-mchip-val {
    font-family: var(--font-disp);
    font-size: 15px; font-weight: 700;
    color: var(--c-txt);
}
.r-mchip-bar {
    height: 3px; border-radius: 2px;
    background: rgba(255,255,255,0.07); overflow: hidden; margin-top: 2px;
}
.r-mchip-bar-fill {
    height: 100%; border-radius: 2px;
    background: var(--c-green);
    opacity: 0.7;
}

/* ── ERROR CARD ── */
.ecard {
    flex: 1;
    border: 1px solid rgba(248,113,113,0.2);
    border-radius: 18px;
    background: var(--c-danger-bg);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 8px; min-height: 0;
    text-align: center; padding: 24px;
}
.e-icon { font-size: 32px; }
.e-title { font-family: var(--font-disp); font-size: 16px; font-weight: 700; color: var(--c-danger); }
.e-body { font-size: 12px; color: rgba(255,255,255,0.5); line-height: 1.7; }

/* ── FOOTER ── */
.foot {
    flex-shrink: 0;
    height: 34px;
    background: var(--c-bg2);
    border-top: 1px solid var(--c-border);
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 24px;
}
.foot-txt {
    font-size: 10px;
    color: var(--c-txt3);
    letter-spacing: 0.2px;
}
.foot-txt strong { color: var(--c-amber); font-weight: 600; }
.foot-models {
    display: flex; gap: 5px;
}
.foot-model-tag {
    font-size: 9px; font-weight: 600;
    letter-spacing: 0.3px;
    padding: 2px 8px; border-radius: 99px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--c-border);
    color: rgba(255,255,255,0.3);
}

/* image override */
[data-testid="stImage"] {
    width: 100% !important;
    height: 100% !important;
}
[data-testid="stImage"] img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    border-radius: 0 !important;
}

/* spinner */
[data-testid="stSpinner"] > div { border-top-color: var(--c-amber) !important; }

/* ══════════════════════════════════
   MOBILE  (≤ 768px)
═══════════════════════════════════ */
@media (max-width: 768px) {
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section.main,
    .block-container, .shell {
        height: auto !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    .nav { height: auto; padding: 10px 14px; flex-wrap: wrap; gap: 6px; }
    .nav-tags { flex-wrap: wrap; }

    .body {
        grid-template-columns: 1fr !important;
        height: auto !important;
        overflow: visible !important;
    }
    .panel {
        padding: 14px 14px;
        overflow: visible !important;
    }
    .panel-left {
        border-right: none !important;
        border-bottom: 1px solid var(--c-border) !important;
    }
    .img-wrap { min-height: 200px; }
    .r-model-grid { grid-template-columns: 1fr 1fr !important; }
    .r-variety { font-size: 28px !important; }
    .r-conf-num { font-size: 26px !important; }

    .foot {
        height: auto; padding: 10px 14px;
        flex-direction: column; gap: 4px; align-items: flex-start;
    }
    .foot-models { flex-wrap: wrap; }
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
# RENDER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    status_cls  = "status-on"  if engine_ok else "status-off"
    status_txt  = "ONLINE"     if engine_ok else "OFFLINE"
    status_dot  = '<span class="blink"></span>' if engine_ok else "✕"

    # ── NAVBAR ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="shell">
      <nav class="nav">
        <div class="nav-brand">
          <span class="nav-brand-icon">🥭</span>
          <span class="nav-brand-name">MangoLeaf<em>VarietyBD</em></span>
        </div>
        <div class="nav-tags">
          <span class="nav-tag">Hybrid Ensemble ×4</span>
          <span class="nav-tag">MangoLeafVarietyBD Dataset</span>
          <span class="nav-tag {status_cls}">{status_dot} {status_txt}</span>
        </div>
      </nav>
      <div class="body">
    """, unsafe_allow_html=True)

    # ── LEFT PANEL ────────────────────────────────────────────────────────────
    st.markdown('<div class="panel panel-left">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Input Sample</div>', unsafe_allow_html=True)

    # File uploader (compact)
    uploaded = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    # Image preview area
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="img-wrap">
          <div class="img-placeholder">
            <div class="img-placeholder-icon">🍃</div>
            <div class="img-placeholder-text">
              No image loaded<br>
              <span style="opacity:.5;font-size:11px">JPG · JPEG · PNG</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Info chips
    st.markdown("""
    <div class="chips-row">
      <span class="chip"><span class="chip-dot"></span>224×224 resize</span>
      <span class="chip"><span class="chip-dot"></span>4-model ensemble</span>
      <span class="chip"><span class="chip-dot"></span>65% threshold</span>
    </div>
    """, unsafe_allow_html=True)

    # Analyse button
    analyse_clicked = False
    if uploaded:
        analyse_clicked = st.button("🔬  Analyse Leaf", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)  # /panel-left

    # ── RIGHT PANEL ───────────────────────────────────────────────────────────
    st.markdown('<div class="panel panel-right">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Detection Result</div>', unsafe_allow_html=True)

    if analyse_clicked and uploaded:
        if not engine_ok:
            st.markdown("""
            <div class="ecard">
              <div class="e-icon">⚠️</div>
              <div class="e-title">Engine Offline</div>
              <div class="e-body">
                <code>Hybrid-MangoLeaf_bundle.pth</code> not found.<br>
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
            <div class="ecard">
              <div class="e-icon">❌</div>
              <div class="e-title">Below Confidence Threshold</div>
              <div class="e-body">
                Score {score:.1f}% &lt; {THRESHOLD:.0f}% threshold.<br>
                This may not be a recognisable mango leaf.<br>
                Please upload a clear image on a plain background.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Build model chips
            chips_html = ""
            for mname, mconf in per_model.items():
                bar_w = min(mconf, 100)
                chips_html += f"""
                <div class="r-mchip">
                  <span class="r-mchip-name">{mname}</span>
                  <span class="r-mchip-val">{mconf:.1f}%</span>
                  <div class="r-mchip-bar">
                    <div class="r-mchip-bar-fill" style="width:{bar_w:.1f}%"></div>
                  </div>
                </div>"""

            st.markdown(f"""
            <div class="rcard">
              <div class="r-tag"><span class="r-dot"></span>Variety Identified</div>
              <div class="r-variety">{variety}</div>
              <div class="r-conf-row">
                <span class="r-conf-num">{score:.1f}%</span>
                <span class="r-conf-lbl">ensemble confidence</span>
              </div>
              <div class="r-bar-track">
                <div class="r-bar-fill" style="width:{min(score,100):.1f}%"></div>
              </div>
              <div class="r-divider"></div>
              <div class="r-models-label">Per-Model Scores</div>
              <div class="r-model-grid">{chips_html}</div>
            </div>
            """, unsafe_allow_html=True)

            if score > 85:
                st.balloons()
    else:
        st.markdown("""
        <div class="idle-box">
          <div class="idle-glyph">🍃</div>
          <div class="idle-msg">
            Neural engine idle<br>
            Upload a leaf sample and click <strong style="color:rgba(255,255,255,.4)">Analyse</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # /panel-right

    # ── CLOSE BODY + FOOTER ───────────────────────────────────────────────────
    st.markdown("""
      </div><!-- /body -->
      <footer class="foot">
        <span class="foot-txt">
          Developed by <strong>Habibur Rahman Sajal</strong>
          &nbsp;·&nbsp; MangoLeafVarietyBD Dataset
        </span>
        <div class="foot-models">
          <span class="foot-model-tag">EfficientNetB0</span>
          <span class="foot-model-tag">MobileNetV2</span>
          <span class="foot-model-tag">DeiT-Tiny</span>
          <span class="foot-model-tag">Swin-Tiny</span>
        </div>
      </footer>
    </div><!-- /shell -->
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
