import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import timm

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MangoLeafVarietyBD",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL CSS  —  only style what Streamlit actually renders, never fight layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap');

/* ── variables ── */
:root{
  --bg:#060a07; --bg2:#0c1510; --bg3:#111c13; --bg4:#1a2b1e;
  --green:#22c55e; --gd:#15803d; --amber:#f59e0b;
  --bdr:rgba(255,255,255,.07); --txt:#f0fdf4;
  --dim:#4a7a5a; --danger:#f87171;
}

/* ── kill chrome ── */
#MainMenu,footer,header[data-testid="stHeader"],
[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebar"]{display:none!important}

/* ── page background ── */
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important}
[data-testid="stAppViewContainer"]>section.main{padding-top:0!important;padding-bottom:0!important}
.block-container{padding-top:0!important;padding-bottom:0!important;max-width:100%!important}

/* ── file uploader ── */
[data-testid="stFileUploader"]>label{display:none!important}
[data-testid="stFileUploader"] section{
  background:var(--bg3)!important;
  border:1.5px dashed rgba(245,158,11,.4)!important;
  border-radius:10px!important;padding:10px 14px!important}
[data-testid="stFileUploader"] section:hover{
  border-color:var(--amber)!important;background:rgba(245,158,11,.05)!important}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small{
  color:rgba(255,255,255,.5)!important;font-size:12px!important;font-family:'DM Sans',sans-serif!important}
[data-testid="stFileUploader"] button{
  background:rgba(245,158,11,.15)!important;color:var(--amber)!important;
  border:1px solid rgba(245,158,11,.35)!important;border-radius:7px!important;
  font-weight:600!important;font-size:11px!important;padding:4px 12px!important}

/* ── image ── */
[data-testid="stImage"] img{
  border-radius:10px!important;width:100%!important;
  max-height:260px!important;object-fit:cover!important}

/* ── button ── */
.stButton>button{
  background:linear-gradient(135deg,var(--gd),#166534)!important;
  color:#fff!important;border:none!important;border-radius:9px!important;
  padding:11px 0!important;font-family:'Syne',sans-serif!important;
  font-size:13px!important;font-weight:700!important;width:100%!important;
  box-shadow:0 4px 16px rgba(21,128,61,.3)!important;
  letter-spacing:.5px!important;transition:all .15s!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 22px rgba(21,128,61,.42)!important}

/* ── spinner ── */
[data-testid="stSpinner"]>div{border-top-color:var(--amber)!important}

/* ── custom html cards ── */
body *{box-sizing:border-box}

.mango-nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 24px;height:52px;
  background:linear-gradient(90deg,#0a2010,#091509 50%,var(--bg));
  border-bottom:1px solid var(--bdr);
  position:relative;
}
.mango-nav::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,var(--gd),transparent 55%);
}
.nav-brand{display:flex;align-items:center;gap:9px}
.nav-logo{font-size:20px}
.nav-title{font-family:'Syne',sans-serif;font-size:16px;font-weight:800;color:#fff;letter-spacing:-.3px}
.nav-title span{color:var(--amber)}
.nav-right{display:flex;gap:6px;align-items:center}
.tag{font-size:10px;font-weight:600;letter-spacing:.5px;padding:3px 10px;border-radius:99px;
  border:1px solid rgba(255,255,255,.1);color:rgba(255,255,255,.45);background:rgba(255,255,255,.04)}
.tag-live{border-color:rgba(34,197,94,.4)!important;background:rgba(34,197,94,.1)!important;
  color:var(--green)!important;display:flex;align-items:center;gap:5px}
.tag-off{border-color:rgba(248,113,113,.4)!important;background:rgba(248,113,113,.1)!important;color:var(--danger)!important}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
.pulse-dot{width:5px;height:5px;border-radius:50%;background:var(--green);
  box-shadow:0 0 6px var(--green);animation:pulse 2s ease-in-out infinite;display:inline-block}

.col-head{
  font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--dim);font-family:'DM Sans',sans-serif;
  display:flex;align-items:center;gap:8px;margin-bottom:8px;
}
.col-head::after{content:'';flex:1;height:1px;background:var(--bdr)}

.no-img{
  min-height:200px;border:1px solid var(--bdr);border-radius:10px;
  background:var(--bg3);display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;margin:6px 0;
}
.no-img-icon{width:48px;height:48px;border-radius:50%;
  background:rgba(34,197,94,.07);border:1px dashed rgba(34,197,94,.2);
  display:flex;align-items:center;justify-content:center;font-size:21px}
.no-img-txt{font-size:11px;color:var(--dim);text-align:center;
  line-height:1.6;font-family:'DM Sans',sans-serif}

.chips{display:flex;gap:5px;flex-wrap:wrap;margin:6px 0}
.chip{font-size:10px;font-weight:600;padding:3px 9px;border-radius:99px;
  border:1px solid var(--bdr);color:rgba(255,255,255,.3);
  background:rgba(255,255,255,.02);font-family:'DM Sans',sans-serif;
  display:flex;align-items:center;gap:4px}
.cdot{width:4px;height:4px;border-radius:50%;background:var(--amber)}

.idle{
  min-height:200px;border:1px dashed var(--bdr);border-radius:14px;
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;
}
.idle-ico{font-size:32px;opacity:.18}
.idle-txt{font-size:12px;color:var(--dim);text-align:center;
  line-height:1.8;font-family:'DM Sans',sans-serif}

.rcard{
  min-height:200px;background:var(--bg3);border:1px solid var(--bdr);
  border-radius:14px;padding:20px 22px;
  display:flex;flex-direction:column;gap:11px;position:relative;overflow:hidden;
}
.rcard::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--gd),var(--amber),transparent 75%);
}
.r-badge{display:flex;align-items:center;gap:5px;font-size:9px;font-weight:700;
  letter-spacing:1.8px;text-transform:uppercase;color:var(--green);font-family:'DM Sans',sans-serif}
.r-bdot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 7px var(--green)}
.r-variety{font-family:'Syne',sans-serif;font-size:clamp(24px,2.8vw,38px);
  font-weight:800;color:#fff;line-height:1.05;letter-spacing:-.5px}
.r-crow{display:flex;align-items:baseline;gap:8px}
.r-cnum{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--amber)}
.r-clbl{font-size:12px;color:var(--dim);font-family:'DM Sans',sans-serif}
.r-bar{height:5px;border-radius:3px;background:rgba(255,255,255,.06);overflow:hidden}
.r-barfill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--gd),var(--amber))}
.r-div{height:1px;background:var(--bdr)}
.r-mlbl{font-size:9px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;
  color:var(--dim);font-family:'DM Sans',sans-serif}
.r-mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}
.r-mc{background:var(--bg4);border:1px solid var(--bdr);border-radius:8px;padding:8px 9px;display:flex;flex-direction:column;gap:3px}
.r-mcn{font-size:9px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:var(--dim);font-family:'DM Sans',sans-serif}
.r-mcv{font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:var(--txt)}
.r-mcbar{height:3px;border-radius:2px;background:rgba(255,255,255,.07);overflow:hidden;margin-top:2px}
.r-mcbf{height:100%;border-radius:2px;background:var(--green);opacity:.6}

.ecard{
  min-height:200px;border:1px solid rgba(248,113,113,.2);border-radius:14px;
  background:rgba(248,113,113,.06);display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:8px;text-align:center;padding:22px;
}
.eico{font-size:28px}
.etitle{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:var(--danger)}
.ebody{font-size:12px;color:rgba(255,255,255,.45);line-height:1.7;font-family:'DM Sans',sans-serif}

.mango-foot{
  height:34px;background:var(--bg2);border-top:1px solid var(--bdr);
  display:flex;align-items:center;justify-content:space-between;padding:0 24px;
  margin-top:8px;
}
.ft{font-size:10px;color:var(--dim);font-family:'DM Sans',sans-serif}
.ft strong{color:var(--amber);font-weight:600}
.ftags{display:flex;gap:4px}
.ftag{font-size:9px;font-weight:600;padding:2px 7px;border-radius:99px;
  background:rgba(255,255,255,.03);border:1px solid var(--bdr);
  color:rgba(255,255,255,.25);font-family:'DM Sans',sans-serif}

/* ── mobile ── */
@media(max-width:720px){
  .mango-nav{height:auto;padding:10px 14px;flex-wrap:wrap;gap:6px}
  .nav-right{flex-wrap:wrap}
  .r-mgrid{grid-template-columns:1fr 1fr!important}
  .mango-foot{height:auto;padding:8px 14px;flex-direction:column;align-items:flex-start;gap:4px}
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_engine():
    p = "Hybrid-MangoLeaf_bundle.pth"
    if not os.path.exists(p):
        return None, None
    bundle = torch.load(p, map_location="cpu")
    nc = len(bundle["classes"])
    loaded = {}
    for name in bundle["model_names"]:
        if name == "EfficientNetB0":
            m = models.efficientnet_b0(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(0.2),
                nn.Linear(m.classifier[1].in_features, nc))
        elif name == "MobileNetV2":
            m = models.mobilenet_v2(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(0.2),
                nn.Linear(m.classifier[1].in_features, nc))
        elif name == "DeiT-Tiny":
            m = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=nc)
        elif name == "Swin-Tiny":
            m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=nc)
        else:
            continue
        if name in bundle["states"]:
            m.load_state_dict(bundle["states"][name])
        m.eval()
        loaded[name] = m
    return loaded, bundle

engine, meta = load_engine()
OK = engine is not None

THRESHOLD = 80.0
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def infer(img):
    t = TF(img).unsqueeze(0)
    pm, ps = {}, []
    with torch.no_grad():
        for n, m in engine.items():
            p = F.softmax(m(t), dim=1)
            w = meta["weights"].get(n, 0.0)
            pm[n] = float(torch.max(p).item()) * 100
            ps.append(p * w)
        fp = torch.stack(ps).sum(0)
        conf, idx = torch.max(fp, 1)
    return meta["classes"][idx.item()], float(conf.item()) * 100, pm


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
# NAV
st.markdown(f"""
<div class="mango-nav">
  <div class="nav-brand">
    <span class="nav-logo">🥭</span>
    <span class="nav-title">MangoLeaf<span>VarietyBD</span></span>
  </div>
  <div class="nav-right">
    <span class="tag">Hybrid Ensemble ×4</span>
    <span class="tag">MangoLeafVarietyBD Dataset</span>
    <span class="tag {'tag-live' if OK else 'tag-off'}">
      {'<span class="pulse-dot"></span>' if OK else '✕'} {'ONLINE' if OK else 'OFFLINE'}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TWO COLUMNS — the ONLY reliable layout in Streamlit ──
left, right = st.columns([5, 7], gap="small")

with left:
    st.markdown('<div class="col-head">Input Sample</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "leaf", type=["jpg","jpeg","png"], label_visibility="collapsed"
    )

    img = None
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)
    else:
        st.markdown("""
        <div class="no-img">
          <div class="no-img-icon">🍃</div>
          <div class="no-img-txt">No image loaded<br>
            <span style="opacity:.45;font-size:10px">JPG · JPEG · PNG</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="chips">
      <span class="chip"><span class="cdot"></span>224×224 input</span>
      <span class="chip"><span class="cdot"></span>4-model ensemble</span>
      <span class="chip"><span class="cdot"></span>80% threshold</span>
    </div>""", unsafe_allow_html=True)

    clicked = False
    if img is not None:
        clicked = st.button("🔬  Analyse Leaf", use_container_width=True)

with right:
    st.markdown('<div class="col-head">Detection Result</div>', unsafe_allow_html=True)

    # run inference
    if clicked and img is not None:
        if not OK:
            st.session_state["res"] = "offline"
        else:
            with st.spinner("Running ensemble inference…"):
                v, s, pm = infer(img)
            st.session_state["res"] = (v, s, pm)

    if img is None:
        st.session_state.pop("res", None)

    res = st.session_state.get("res", None)

    if res is None:
        st.markdown("""
        <div class="idle">
          <div class="idle-ico">🍃</div>
          <div class="idle-txt">Neural engine idle<br>
            Upload a sample &amp; click <b style="color:rgba(255,255,255,.3)">Analyse Leaf</b>
          </div>
        </div>""", unsafe_allow_html=True)

    elif res == "offline":
        st.markdown("""
        <div class="ecard">
          <div class="eico">⚠️</div>
          <div class="etitle">Bundle Not Found</div>
          <div class="ebody">Hybrid-MangoLeaf_bundle.pth missing.<br>
            Place it in the same folder as app.py.</div>
        </div>""", unsafe_allow_html=True)

    else:
        variety, score, pm = res
        if score < THRESHOLD:
            st.markdown(f"""
            <div class="ecard">
              <div class="eico">❌</div>
              <div class="etitle">Low Confidence — {score:.1f}%</div>
              <div class="ebody">Below the {THRESHOLD:.0f}% threshold.<br>
                Upload a clearer mango leaf on a plain background.</div>
            </div>""", unsafe_allow_html=True)
        else:
            mc = "".join(f"""
            <div class="r-mc">
              <span class="r-mcn">{n}</span>
              <span class="r-mcv">{c:.1f}%</span>
              <div class="r-mcbar"><div class="r-mcbf" style="width:{min(c,100):.1f}%"></div></div>
            </div>""" for n, c in pm.items())

            st.markdown(f"""
            <div class="rcard">
              <div class="r-badge"><span class="r-bdot"></span>Variety Identified</div>
              <div class="r-variety">{variety}</div>
              <div class="r-crow">
                <span class="r-cnum">{score:.1f}%</span>
                <span class="r-clbl">ensemble confidence</span>
              </div>
              <div class="r-bar"><div class="r-barfill" style="width:{min(score,100):.1f}%"></div></div>
              <div class="r-div"></div>
              <div class="r-mlbl">Per-Model Scores</div>
              <div class="r-mgrid">{mc}</div>
            </div>""", unsafe_allow_html=True)

            if score > 85:
                st.balloons()

# FOOTER
st.markdown("""
<div class="mango-foot">
  <span class="ft">Developed by <strong>Habibur Rahman Sajal</strong> &nbsp;·&nbsp; MangoLeafVarietyBD Dataset</span>
  <div class="ftags">
    <span class="ftag">EfficientNetB0</span>
    <span class="ftag">MobileNetV2</span>
    <span class="ftag">DeiT-Tiny</span>
    <span class="ftag">Swin-Tiny</span>
  </div>
</div>
""", unsafe_allow_html=True)
