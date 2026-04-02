import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import timm
import time

# ---------------------------------------------------------
# STAGE 1: ULTRA-PREMIUM UI & HIGH-CONTRAST DARK THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="MangoLeafVarietyBD Elite",
    page_icon="🥭",
    layout="wide"
)

# Professional CSS for Visibility Fixes & Spacing
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Root Background: Deep Space Dark */
    .stApp {
        background-color: #020617 !important;
        color: #ffffff !important;
    }

    /* Fix Top Header Bar Visibility */
    header[data-testid="stHeader"] {
        background: rgba(2, 6, 23, 0.9) !important;
        backdrop-filter: blur(12px);
    }

    /* SIDEBAR: Spacing & High-Contrast Visibility Fix */
    [data-testid="stSidebar"] {
        background-color: #030712 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-label {
        color: #94a3b8 !important; /* Vivid Grey-Blue Label */
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 35px !important;
        margin-bottom: 10px !important;
        display: block;
    }

    /* FIXED: Sidebar value text made PURE WHITE & BOLD */
    .sidebar-value {
        color: #ffffff !important; 
        font-size: 1.15rem;
        font-weight: 800;
        background: rgba(255, 255, 255, 0.08);
        padding: 12px 18px;
        border-radius: 12px;
        display: block;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: inset 0 0 10px rgba(0,0,0,0.3);
    }

    /* FILE UPLOADER: High Contrast & Button Fix */
    [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 2px dashed #10b981 !important;
        border-radius: 20px !important;
        padding: 30px !important;
    }
    
    /* FIXED: Upload Button Text (Deep Dark on Pure White) */
    [data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #020617 !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 10px 25px !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
    }
    
    /* FIXED: Small hint text (200MB, JPG etc) made CLEAR WHITE */
    [data-testid="stFileUploader"] div div {
        color: #ffffff !important; 
        font-weight: 700 !important;
        font-size: 1rem !important;
        opacity: 1 !important;
    }

    /* MAIN CARDS: Floating Glassmorphism */
    .elite-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 32px;
        padding: 40px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.6);
        backdrop-filter: blur(20px);
    }

    /* TITLES: Radiant Emerald Gradients */
    .hero-title {
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #4ade80 0%, #2dd4bf 50%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2.5px;
    }

    .variety-label {
        font-size: 6.5rem;
        font-weight: 800;
        color: #ffffff !important;
        text-shadow: 0 0 40px rgba(16, 185, 129, 0.4);
        margin: 10px 0;
        line-height: 1.1;
    }

    /* ANALYZE BUTTON: Premium Execution */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        border: none;
        padding: 22px;
        border-radius: 18px;
        font-weight: 800;
        font-size: 1.3rem;
        text-transform: uppercase;
        width: 100%;
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.4);
    }
    
    .stProgress div[data-baseweb="progress-bar"] > div {
        background-color: #10b981 !important;
    }

    /* FIXED: General help/info text made visible */
    .stMarkdown p {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# STAGE 2: HYBRID ENSEMBLE ENGINE LOADER
# ---------------------------------------------------------
@st.cache_resource
def load_elite_hybrid_engine():
    bundle_path = 'Hybrid-MangoLeaf_bundle.pth'
    if not os.path.exists(bundle_path): return None, None
    
    bundle = torch.load(bundle_path, map_location=torch.device('cpu'))
    num_classes = len(bundle['classes'])
    models_dict = {}
    
    for name in bundle['model_names']:
        if name == "EfficientNetB0":
            m = models.efficientnet_b0(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, num_classes))
        elif name == "MobileNetV2":
            m = models.mobilenet_v2(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, num_classes))
        elif name == "DeiT-Tiny":
            m = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
        elif name == "Swin-Tiny":
            m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        
        if name in bundle['states']: m.load_state_dict(bundle['states'][name])
        m.eval()
        models_dict[name] = m
    return models_dict, bundle

models_engine, bundle_meta = load_elite_hybrid_engine()

# ---------------------------------------------------------
# STAGE 3: SIDEBAR (Elite Spacious Panel)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#10b981; margin-bottom:40px; font-size:2rem; font-weight:800;'>⚡ Elite Engine</h2>", unsafe_allow_html=True)
    
    st.markdown("<p class='sidebar-label'>🏷️ Dataset</p>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-value'>MengoLeafVerityBd</span>", unsafe_allow_html=True)
    
    st.markdown("<p class='sidebar-label'>👨‍💻 Lead Developer</p>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-value'>Habibur Rahman Sajal</span>", unsafe_allow_html=True)
    
    st.markdown("<p class='sidebar-label'>🧬 Architecture</p>", unsafe_allow_html=True)
    st.markdown("<span class='sidebar-value'>Hybrid Ensemble x4</span>", unsafe_allow_html=True)
    
    st.write("<br>"*2, unsafe_allow_html=True)
    
    if models_engine: 
        st.markdown("<div style='background:rgba(16,185,129,0.15); color:#10b981; padding:15px; border-radius:15px; text-align:center; font-weight:800; border:1px solid #10b981; box-shadow: 0 4px 20px rgba(0,0,0,0.4);'>🟢 SYSTEM ONLINE</div>", unsafe_allow_html=True)
    else: 
        st.error("🔴 BUNDLE MISSING")
    
    st.write("<br>"*4, unsafe_allow_html=True)
    st.caption("Release v4.5 Elite Build | © 2026")

# ---------------------------------------------------------
# STAGE 4: MAIN INTERFACE & VALIDATION LOGIC
# ---------------------------------------------------------
def main():
    st.markdown("<h1 class='hero-title'>MangoLeafVarietyBD</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8; font-size:1.4rem; font-weight:600; margin-top:-20px;'>Advanced Hybrid Intelligence for Agricultural Precision</p>", unsafe_allow_html=True)
    
    st.write("<br><br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1, 1.2], gap="large")

    with col_l:
        st.markdown("<div class='elite-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:white; margin-bottom:20px;'>📥 Data Input</h3>", unsafe_allow_html=True)
        file = st.file_uploader("Drop mango leaf sample here", type=['jpg', 'jpeg', 'png'])
        
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True, caption="Sample Ready for Neural Analysis")
            
            if st.button("🚀 Execute Diagnosis"):
                if models_engine:
                    with st.spinner("Decoding Spectral Features..."):
                        tf = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        tensor = tf(img).unsqueeze(0)
                        all_p = []
                        with torch.no_grad():
                            for n, m in models_engine.items():
                                p = F.softmax(m(tensor), dim=1)
                                all_p.append(p * bundle_meta['weights'].get(n, 0.0))
                            final_p = torch.stack(all_p).sum(dim=0)
                            conf, idx = torch.max(final_p, 1)
                        st.session_state['final_res'] = (bundle_meta['classes'][idx.item()], conf.item() * 100)
                else:
                    st.error("Engine failure: Hybrid bundle not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if 'final_res' in st.session_state:
            variety, score = st.session_state['final_res']
            
            # --- HIGH-CONFIDENCE VALIDATION (65% THRESHOLD) ---
            THRESHOLD = 65.0 
            
            if score < THRESHOLD:
                st.markdown("<div class='elite-card'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#f87171; font-weight:800; letter-spacing:1px;'>❌ INVALID INPUT DETECTED</p>", unsafe_allow_html=True)
                st.markdown("<h2 style='color:white;'>This is not a recognized Mango Leaf!</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color:#ffffff; font-weight:500;'>Our AI has detected a non-target object or high background noise. Please upload a clear image of a <b>single Mango Leaf</b> on a plain background.</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            else:
                st.markdown("<div class='elite-card'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#10b981; font-weight:800; letter-spacing:3px;'>TARGET IDENTIFIED</p>", unsafe_allow_html=True)
                st.markdown(f"<h1 class='variety-label'>{variety}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#ffffff; font-size:1.7rem; font-weight:700;'>{score:.2f}% <span style='color:#94a3b8; font-weight:400;'>Inference Confidence</span></p>", unsafe_allow_html=True)
                
                st.write("<br>", unsafe_allow_html=True)
                st.progress(score / 100)
                st.markdown("</div>", unsafe_allow_html=True)
                if score > 85: st.balloons()
        else:
            st.markdown("""
                <div style='border: 1px dashed rgba(255,255,255,0.1); border-radius:32px; padding:145px; text-align:center; background:rgba(255,255,255,0.01);'>
                    <p style='color:#94a3b8; font-size:1.3rem; font-weight:500;'>Neural Engine Idle...<br>Scan a sample to generate report.</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()