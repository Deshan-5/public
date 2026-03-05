import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math, time

from core.distributions import *
from tools.stats_tools import *

st.set_page_config(
    page_title="Probability Lab",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page"     not in st.session_state: st.session_state.page     = "landing"
if "lab_type" not in st.session_state: st.session_state.lab_type = "discrete"

# ── handle ?go= tile clicks ──────────────────────────────────────────────────
_qp = st.query_params
if "go" in _qp:
    _dest = _qp["go"]
    st.query_params.clear()
    if   _dest == "disc":    st.session_state.lab_type = "discrete";   st.session_state.page = "lab"
    elif _dest == "cont":    st.session_state.lab_type = "continuous"; st.session_state.page = "lab"
    elif _dest == "tests":   st.session_state.page = "tests"
    elif _dest == "bayes":   st.session_state.page = "bayes"
    elif _dest == "comb":    st.session_state.page = "combinatorics"
    elif _dest == "ci":      st.session_state.page = "ci"
    elif _dest == "landing": st.session_state.page = "landing"
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# ULTRA GLOBAL CSS + JS PARTICLE ENGINE
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;500;700;900&family=JetBrains+Mono:wght@300;400;600;800&display=swap');

:root {
  --cyan:   #00f5ff;
  --pink:   #ff2d78;
  --amber:  #ffb700;
  --violet: #9d4edd;
  --green:  #00ff9d;
  --bg:     #020408;
  --bg2:    #060d18;
  --bg3:    #0a1628;
  --border: rgba(255,255,255,0.06);
  --glow-cyan:   0 0 20px rgba(0,245,255,0.35), 0 0 60px rgba(0,245,255,0.12);
  --glow-pink:   0 0 20px rgba(255,45,120,0.35), 0 0 60px rgba(255,45,120,0.12);
  --glow-amber:  0 0 20px rgba(255,183,0,0.35),  0 0 60px rgba(255,183,0,0.12);
  --glow-violet: 0 0 20px rgba(157,78,221,0.35), 0 0 60px rgba(157,78,221,0.12);
  --glow-green:  0 0 20px rgba(0,255,157,0.35),  0 0 60px rgba(0,255,157,0.12);
}

/* ─── RESET ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  color: #c8d6e8 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  overflow-x: hidden;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="stHeader"]          { display: none !important; }
.block-container,
[data-testid="stMainBlockContainer"] {
  padding: 0 !important;
  max-width: 100% !important;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
  gap: 0 !important;
}

/* ─── CANVAS PARTICLE BG ────────────────────────────────────── */
#plab-canvas {
  position: fixed; inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.75;
}

/* ─── SCANLINE OVERLAY ───────────────────────────────────────── */
body::after {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.04) 2px,
    rgba(0,0,0,0.04) 4px
  );
  pointer-events: none;
  z-index: 9999;
  animation: scanmove 8s linear infinite;
}
@keyframes scanmove {
  from { background-position: 0 0; }
  to   { background-position: 0 100vh; }
}

/* ─── NOISE GRAIN ────────────────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 9998; opacity: 0.4;
}

/* ─── SLIDERS ────────────────────────────────────────────────── */
[data-testid="stSlider"] label {
  color: rgba(200,214,232,0.45) !important;
  font-size: 10px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, var(--cyan), var(--violet)) !important;
  height: 3px !important;
  border-radius: 2px !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
  background: var(--bg3) !important;
  border: 1px solid rgba(0,245,255,0.4) !important;
  color: var(--cyan) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
}

/* ─── INPUTS ─────────────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"]   input,
textarea {
  background: rgba(6,13,24,0.9) !important;
  border: 1px solid rgba(0,245,255,0.15) !important;
  color: var(--cyan) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
  transition: all 0.3s ease !important;
  backdrop-filter: blur(10px);
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"]   input:focus,
textarea:focus {
  border-color: rgba(0,245,255,0.6) !important;
  box-shadow: 0 0 0 2px rgba(0,245,255,0.12), inset 0 0 20px rgba(0,245,255,0.05) !important;
  outline: none;
}

/* ─── SELECT / RADIO / CHECKBOX ──────────────────────────────── */
[data-testid="stSelectbox"] > div {
  background: rgba(6,13,24,0.9) !important;
  border: 1px solid rgba(0,245,255,0.15) !important;
  border-radius: 8px !important;
  backdrop-filter: blur(10px);
}
[data-testid="stSelectbox"] label,
[data-testid="stRadio"]     label,
[data-testid="stCheckbox"]  label {
  color: rgba(200,214,232,0.5) !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ─── BUTTONS — completely reimagined ───────────────────────── */
[data-testid="stButton"] > button {
  background: transparent !important;
  border: 1px solid rgba(0,245,255,0.25) !important;
  color: var(--cyan) !important;
  font-family: 'Orbitron', monospace !important;
  letter-spacing: 3px !important;
  font-size: 9px !important;
  font-weight: 700 !important;
  border-radius: 6px !important;
  padding: 10px 18px !important;
  min-height: 40px;
  transition: all 0.4s cubic-bezier(0.23,1,0.32,1) !important;
  width: 100%;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  text-transform: uppercase;
}
[data-testid="stButton"] > button::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(0,245,255,0.12), transparent 60%);
  opacity: 0;
  transition: opacity 0.3s ease;
}
[data-testid="stButton"] > button::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  transform: scaleX(0);
  transition: transform 0.4s ease;
}
[data-testid="stButton"] > button:hover {
  border-color: rgba(0,245,255,0.7) !important;
  box-shadow: var(--glow-cyan) !important;
  transform: translateY(-2px) !important;
  color: #fff !important;
}
[data-testid="stButton"] > button:hover::before { opacity: 1; }
[data-testid="stButton"] > button:hover::after  { transform: scaleX(1); }
[data-testid="stButton"] > button:active {
  transform: translateY(0) scale(0.97) !important;
}

/* ─── TABS — floating pill style ────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: rgba(6,13,24,0.7) !important;
  border-radius: 12px !important;
  padding: 5px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  backdrop-filter: blur(20px) !important;
  gap: 4px !important;
  margin-bottom: 4px !important;
}
[data-testid="stTabs"] [role="tab"] {
  color: rgba(200,214,232,0.3) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 1.5px !important;
  border-radius: 8px !important;
  min-height: 36px !important;
  padding: 0 16px !important;
  transition: all 0.3s ease !important;
  cursor: pointer;
  text-transform: uppercase;
}
[data-testid="stTabs"] [role="tab"]:hover { color: rgba(200,214,232,0.7) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(157,78,221,0.1)) !important;
  color: var(--cyan) !important;
  border: 1px solid rgba(0,245,255,0.3) !important;
  box-shadow: 0 0 18px rgba(0,245,255,0.15) !important;
}
[data-testid="stTabs"] [data-testid="stTabContent"] { padding-top: 22px !important; }

/* ─── EXPANDER ───────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: rgba(6,13,24,0.6) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  border-radius: 10px !important;
  backdrop-filter: blur(10px);
}
[data-testid="stExpander"] summary {
  color: rgba(200,214,232,0.4) !important;
  font-size: 10px !important;
  letter-spacing: 2px !important;
  font-family: 'JetBrains Mono', monospace !important;
  min-height: 40px !important;
  cursor: pointer;
}

/* ─── ALERTS ─────────────────────────────────────────────────── */
[data-testid="stSuccess"] { background: rgba(0,255,157,0.06) !important; border: 1px solid rgba(0,255,157,0.3) !important; border-radius: 10px !important; }
[data-testid="stError"]   { background: rgba(255,45,120,0.06) !important; border: 1px solid rgba(255,45,120,0.3) !important; border-radius: 10px !important; }
[data-testid="stInfo"]    { background: rgba(0,245,255,0.05) !important; border: 1px solid rgba(0,245,255,0.25) !important; border-radius: 10px !important; }
[data-testid="stWarning"] { background: rgba(255,183,0,0.06)  !important; border: 1px solid rgba(255,183,0,0.3)  !important; border-radius: 10px !important; }

/* ─── ELEMENT SPACING ────────────────────────────────────────── */
[data-testid="stSlider"]        { margin-bottom: 14px !important; }
[data-testid="stNumberInput"]   { margin-bottom: 10px !important; }
[data-testid="stSelectbox"]     { margin-bottom: 10px !important; }
[data-testid="stCheckbox"]      { margin-bottom: 8px !important; }
[data-testid="stRadio"]         { margin-bottom: 10px !important; }
[data-testid="stTextInput"]     { margin-bottom: 10px !important; }
[data-testid="stTextArea"]      { margin-bottom: 10px !important; }

/* ─── SCROLLBAR ──────────────────────────────────────────────── */
::-webkit-scrollbar            { width: 3px; height: 3px; }
::-webkit-scrollbar-track      { background: transparent; }
::-webkit-scrollbar-thumb      { background: rgba(0,245,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover{ background: rgba(0,245,255,0.4); }

/* ─── KEYFRAMES ──────────────────────────────────────────────── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes pulseOrb {
  0%,100% { box-shadow: 0 0 6px var(--green), 0 0 12px var(--green); opacity: 0.8; }
  50%     { box-shadow: 0 0 12px var(--green), 0 0 30px var(--green), 0 0 50px rgba(0,255,157,0.3); opacity: 1; }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
@keyframes borderFlow {
  0%   { border-color: rgba(0,245,255,0.3); box-shadow: 0 0 0px rgba(0,245,255,0); }
  50%  { border-color: rgba(157,78,221,0.5); box-shadow: 0 0 20px rgba(157,78,221,0.2); }
  100% { border-color: rgba(0,245,255,0.3); box-shadow: 0 0 0px rgba(0,245,255,0); }
}
@keyframes floatY {
  0%,100% { transform: translateY(0px); }
  50%     { transform: translateY(-6px); }
}
@keyframes ticker {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}
@keyframes glitchIn {
  0%  { clip-path: inset(0 100% 0 0); opacity: 0; }
  40% { clip-path: inset(0 20% 0 0);  opacity: 0.8; }
  60% { clip-path: inset(0 35% 0 0);  opacity: 0.6; }
  80% { clip-path: inset(0 5% 0 0);   opacity: 0.9; }
  100%{ clip-path: inset(0 0% 0 0);   opacity: 1; }
}
@keyframes rotateHalo {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes cardReveal {
  from { opacity: 0; transform: translateY(30px) scale(0.96); filter: blur(4px); }
  to   { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}
@keyframes numberCount {
  from { opacity: 0; transform: scale(0.5) rotateX(90deg); }
  to   { opacity: 1; transform: scale(1) rotateX(0deg); }
}
@keyframes lineGrow {
  from { width: 0; }
  to   { width: 100%; }
}
@keyframes hueRotate {
  from { filter: hue-rotate(0deg); }
  to   { filter: hue-rotate(360deg); }
}
@keyframes blink {
  0%,100% { opacity: 1; }
  50%      { opacity: 0; }
}
@keyframes scanLine {
  0%   { top: -4px; opacity: 0.9; }
  100% { top: 110%; opacity: 0; }
}

/* ─── PORTAL CARD HOVER ──────────────────────────────────────── */
a[href="?go=disc"]:hover .portal-card {
  transform: translateY(-6px) scale(1.01) !important;
  border-color: rgba(0,245,255,0.45) !important;
  box-shadow: 0 28px 70px rgba(0,0,0,0.6),0 0 40px rgba(0,245,255,0.18),inset 0 1px 0 rgba(0,245,255,0.2) !important;
}
a[href="?go=cont"]:hover .portal-card {
  transform: translateY(-6px) scale(1.01) !important;
  border-color: rgba(255,183,0,0.45) !important;
  box-shadow: 0 28px 70px rgba(0,0,0,0.6),0 0 40px rgba(255,183,0,0.18),inset 0 1px 0 rgba(255,183,0,0.2) !important;
}

/* ─── TOOLKIT TILE HOVER ─────────────────────────────────────── */
.tool-tile-tests { transition: transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease; }
.tool-tile-tests:hover { transform: translateY(-8px) scale(1.02) !important; border-color: #ff2d7855 !important; box-shadow: 0 28px 70px rgba(0,0,0,0.65),0 0 45px #ff2d7828,inset 0 1px 0 #ff2d7822 !important; }
.tool-tile-comb { transition: transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease; }
.tool-tile-comb:hover { transform: translateY(-8px) scale(1.02) !important; border-color: #9d4edd55 !important; box-shadow: 0 28px 70px rgba(0,0,0,0.65),0 0 45px #9d4edd28,inset 0 1px 0 #9d4edd22 !important; }
.tool-tile-bayes { transition: transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease; }
.tool-tile-bayes:hover { transform: translateY(-8px) scale(1.02) !important; border-color: #00ff9d55 !important; box-shadow: 0 28px 70px rgba(0,0,0,0.65),0 0 45px #00ff9d28,inset 0 1px 0 #00ff9d22 !important; }
.tool-tile-ci { transition: transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease; }
.tool-tile-ci:hover { transform: translateY(-8px) scale(1.02) !important; border-color: #ffb70055 !important; box-shadow: 0 28px 70px rgba(0,0,0,0.65),0 0 45px #ffb70028,inset 0 1px 0 #ffb70022 !important; }

/* ─── CURSOR GLOW TRAIL ──────────────────────────────────────── */
#cursor-glow {
  position: fixed;
  width: 320px; height: 320px;
  border-radius: 50%;
  pointer-events: none;
  z-index: 9997;
  transform: translate(-50%,-50%);
  background: radial-gradient(circle, rgba(0,245,255,0.07) 0%, rgba(157,78,221,0.04) 35%, transparent 70%);
  transition: background 0.4s ease;
  filter: blur(2px);
}
#cursor-dot {
  position: fixed;
  width: 6px; height: 6px;
  border-radius: 50%;
  pointer-events: none;
  z-index: 9997;
  transform: translate(-50%,-50%);
  background: #00f5ff;
  box-shadow: 0 0 8px #00f5ff, 0 0 20px rgba(0,245,255,0.5);
  transition: width 0.2s, height 0.2s, background 0.2s;
  opacity: 0;
}

/* ─── TILE SCAN LINE ─────────────────────────────────────────── */
.scan-wrap { position: relative; overflow: hidden; }
.scan-wrap::after {
  content: '';
  position: absolute;
  left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
  top: -4px;
  opacity: 0;
  pointer-events: none;
  z-index: 10;
}
.scan-wrap:hover::after {
  animation: scanLine 0.55s ease forwards;
}

/* ─── SOUND TOGGLE — HIDDEN ──────────────────────────────────── */
#sound-toggle { display: none !important; }

/* ─── CUSTOM CURSOR — handled by JS injection ────────────────── */

/* ─── PAGE WRAPPER ───────────────────────────────────────────── */
.plab-page { animation: fadeUp 0.5s cubic-bezier(0.23,1,0.32,1) both; }
</style>
""", unsafe_allow_html=True)

# ── SPACE BG + CURSOR via self-contained hidden iframe ────────
st.markdown("""
<iframe id="plab-fx-frame"
  srcdoc="<script>
(function(){
  var doc=window.parent.document;
  var bod=doc.body;
  var PAL=['#00f5ff','#ff2d78','#ffb700','#9d4edd','#00ff9d'];
  function mk(id,tag,s){var e=doc.getElementById(id);if(e)e.remove();e=doc.createElement(tag);e.id=id;Object.assign(e.style,s);bod.appendChild(e);return e;}
  var cnv=mk('plab-canvas','canvas',{position:'fixed',top:'0',left:'0',width:'100vw',height:'100vh',pointerEvents:'none',zIndex:'1'});
  var glw=mk('plab-glow','div',{position:'fixed',width:'360px',height:'360px',borderRadius:'50%',pointerEvents:'none',zIndex:'9',transform:'translate(-50%,-50%)',background:'radial-gradient(circle,rgba(0,245,255,0.08) 0%,transparent 70%)',filter:'blur(6px)',left:'-999px',top:'-999px'});
  var rng=mk('plab-ring','div',{position:'fixed',width:'28px',height:'28px',borderRadius:'50%',border:'1.5px solid rgba(0,245,255,0.8)',pointerEvents:'none',zIndex:'9998',transform:'translate(-50%,-50%)',left:'-999px',top:'-999px',transition:'width 0.1s,height 0.1s,border-color 0.15s'});
  var dot=mk('plab-dot','div',{position:'fixed',width:'5px',height:'5px',borderRadius:'50%',background:'#00f5ff',boxShadow:'0 0 8px #00f5ff,0 0 20px rgba(0,245,255,0.6)',pointerEvents:'none',zIndex:'9999',transform:'translate(-50%,-50%)',left:'-999px',top:'-999px'});
  var cch=mk('plab-ch','div',{position:'fixed',width:'14px',height:'1.5px',background:'rgba(0,245,255,0.85)',pointerEvents:'none',zIndex:'9999',transform:'translate(-50%,-50%)',left:'-999px',top:'-999px'});
  var ccv=mk('plab-cv','div',{position:'fixed',width:'1.5px',height:'14px',background:'rgba(0,245,255,0.85)',pointerEvents:'none',zIndex:'9999',transform:'translate(-50%,-50%)',left:'-999px',top:'-999px'});
  var pfl=mk('plab-fl','div',{position:'fixed',inset:'0',background:'#00f5ff',opacity:'0',pointerEvents:'none',zIndex:'99999',transition:'opacity 0.18s ease'});
  var sc=doc.getElementById('plab-cs');if(!sc){sc=doc.createElement('style');sc.id='plab-cs';doc.head.appendChild(sc);}
  sc.textContent='body,canvas,[data-testid=stApp],[data-testid=stMainBlockContainer]{cursor:none!important}button,a,input,select,textarea,label,[role=button],[role=slider]{cursor:none!important}';
  var mx=-999,my=-999,gx=-999,gy=-999,rx=-999,ry=-999,ci=0,ct=0;
  function sp(e,x,y){e.style.left=x+'px';e.style.top=y+'px';}
  doc.addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY;sp(dot,mx,my);sp(cch,mx,my);sp(ccv,mx,my);});
  doc.addEventListener('mousedown',function(){rng.style.width='14px';rng.style.height='14px';dot.style.transform='translate(-50%,-50%) scale(2)';});
  doc.addEventListener('mouseup',function(){rng.style.width='28px';rng.style.height='28px';dot.style.transform='translate(-50%,-50%) scale(1)';});
  (function aC(){gx+=(mx-gx)*0.09;gy+=(my-gy)*0.09;rx+=(mx-rx)*0.2;ry+=(my-ry)*0.2;sp(glw,gx,gy);sp(rng,rx,ry);ct++;if(ct>100){ct=0;ci=(ci+1)%5;}var c=PAL[ci];dot.style.background=c;dot.style.boxShadow='0 0 7px '+c+',0 0 20px '+c+'99';cch.style.background=c+'cc';ccv.style.background=c+'cc';rng.style.borderColor=c+'aa';requestAnimationFrame(aC);})();
  var ctx=cnv.getContext('2d'),W=0,H=0;
  function rsz(){W=cnv.width=window.parent.innerWidth;H=cnv.height=window.parent.innerHeight;}rsz();window.parent.addEventListener('resize',rsz);
  var st2=[];for(var i=0;i<280;i++)st2.push({x:Math.random()*5000,y:Math.random()*4000,r:0.3+Math.random()*1.7,a:0.15+Math.random()*0.85,tw:Math.random()*6.28,sp:0.008+Math.random()*0.025});
  var nb=[];for(var i=0;i<7;i++)nb.push({x:Math.random()*5000,y:Math.random()*4000,r:200+Math.random()*300,c:PAL[i%5],a:0.035+Math.random()*0.04});
  var pts=[];for(var i=0;i<120;i++)pts.push({x:Math.random()*5000,y:Math.random()*4000,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,r:0.6+Math.random()*2,c:PAL[~~(Math.random()*5)],a:0.1+Math.random()*0.38,lf:Math.random()*500,mx:400+Math.random()*600});
  var SC=['σ','μ','∫','π','Σ','λ','∞','Δ','φ','β','±','≈','P(X)','E[X]','nCr','0.5','e','≠'];
  var sy=[];for(var i=0;i<70;i++)sy.push({x:Math.random()*5000,y:Math.random()*4000,vy:0.18+Math.random()*0.5,s:SC[~~(Math.random()*SC.length)],c:PAL[~~(Math.random()*5)],a:0.07+Math.random()*0.2,sz:10+Math.random()*13,dr:(Math.random()-.5)*0.1});
  var bl=[];for(var i=0;i<10;i++)bl.push({x:Math.random()*5000,y:Math.random()*4000,vy:-(0.07+Math.random()*0.18),sc:55+Math.random()*110,c:PAL[i%5],a:0,ma:0.07+Math.random()*0.1,fi:true});
  var DL=['P(A|B)','E[X²]','N(μ,σ)','Var(X)','P(∩)','nCr','CLT','σ²','Bayes'];
  var DD={1:[[0,0]],2:[[-1,-1],[1,1]],3:[[-1,-1],[0,0],[1,1]],4:[[-1,-1],[1,-1],[-1,1],[1,1]],5:[[-1,-1],[1,-1],[0,0],[-1,1],[1,1]],6:[[-1,-1],[1,-1],[-1,0],[1,0],[-1,1],[1,1]]};
  var NM=['0.5','∞','π','e','σ','μ'];var OP=['+','−','×','=','~','∈'];
  var ob=[];
  DL.forEach(function(l,i){ob.push({t:'c',l:l,col:PAL[i%5],x:Math.random()*5000,y:Math.random()*4000,vx:(Math.random()-.5)*0.2,vy:(Math.random()-.5)*0.2,r:Math.random()*6.28,vr:(Math.random()-.5)*0.004,sc:0.7+Math.random()*0.55,ph:Math.random()*6.28});});
  for(var f=1;f<=6;f++)ob.push({t:'d',f:f,x:Math.random()*5000,y:Math.random()*4000,vx:(Math.random()-.5)*0.18,vy:(Math.random()-.5)*0.18,r:Math.random()*6.28,vr:(Math.random()-.5)*0.005,sc:0.55+Math.random()*0.5,ph:Math.random()*6.28});
  NM.forEach(function(v,i){ob.push({t:'n',v:v,col:PAL[i%5],x:Math.random()*5000,y:Math.random()*4000,vx:(Math.random()-.5)*0.16,vy:(Math.random()-.5)*0.16,r:0,vr:0,sc:0.9+Math.random()*0.7,ph:Math.random()*6.28});});
  OP.forEach(function(v,i){ob.push({t:'o',v:v,col:PAL[i%5],x:Math.random()*5000,y:Math.random()*4000,vx:(Math.random()-.5)*0.25,vy:(Math.random()-.5)*0.25,r:0,vr:0,sc:1.0+Math.random()*0.8,ph:Math.random()*6.28});});
  var wt=0,gt=0,tt=0;
  function dO(o){var p=0.65+0.35*Math.sin(tt*0.85+o.ph);ctx.save();ctx.translate(o.x,o.y);ctx.rotate(o.r);ctx.scale(o.sc,o.sc);if(o.t==='c'){var w=92,h=52;ctx.beginPath();if(ctx.roundRect)ctx.roundRect(-w/2,-h/2,w,h,9);else ctx.rect(-w/2,-h/2,w,h);ctx.strokeStyle=o.col;ctx.lineWidth=1.5;ctx.globalAlpha=0.3*p;ctx.stroke();ctx.fillStyle=o.col+'22';ctx.globalAlpha=0.18*p;ctx.fill();ctx.font='bold 12px monospace';ctx.fillStyle=o.col;ctx.textAlign='center';ctx.textBaseline='middle';ctx.globalAlpha=0.38*p;ctx.fillText(o.l,0,0);}else if(o.t==='d'){var s=33;ctx.beginPath();if(ctx.roundRect)ctx.roundRect(-s,-s,s*2,s*2,7);else ctx.rect(-s,-s,s*2,s*2);ctx.strokeStyle='rgba(180,210,240,0.75)';ctx.lineWidth=1.4;ctx.globalAlpha=0.32*p;ctx.stroke();ctx.fillStyle='rgba(4,10,22,0.65)';ctx.globalAlpha=0.22*p;ctx.fill();DD[o.f].forEach(function(d){ctx.beginPath();ctx.arc(d[0]*10,d[1]*10,3.5,0,6.28);ctx.fillStyle='rgba(220,235,255,1)';ctx.globalAlpha=0.45*p;ctx.fill();});}else if(o.t==='n'){ctx.font='900 40px monospace';ctx.fillStyle=o.col;ctx.textAlign='center';ctx.textBaseline='middle';ctx.globalAlpha=0.32*p;ctx.fillText(o.v,0,0);}else{ctx.font='900 34px monospace';ctx.fillStyle=o.col;ctx.textAlign='center';ctx.textBaseline='middle';ctx.globalAlpha=0.26*p;ctx.fillText(o.v,0,0);}ctx.restore();}
  function loop(){
    ctx.clearRect(0,0,W,H);
    nb.forEach(function(n){var g=ctx.createRadialGradient(n.x%W,n.y%H,0,n.x%W,n.y%H,n.r);g.addColorStop(0,n.c+'1a');g.addColorStop(1,'transparent');ctx.beginPath();ctx.arc(n.x%W,n.y%H,n.r,0,6.28);ctx.fillStyle=g;ctx.globalAlpha=n.a;ctx.fill();});
    st2.forEach(function(s){s.tw+=s.sp;var a=s.a*(0.35+0.65*Math.abs(Math.sin(s.tw)));ctx.beginPath();ctx.arc(s.x%W,s.y%H,s.r,0,6.28);ctx.fillStyle='#fff';ctx.globalAlpha=a;ctx.fill();});
    ctx.save();ctx.translate(W/2,H/2);ctx.rotate(gt);ctx.strokeStyle='#00f5ff';ctx.globalAlpha=0.02;ctx.lineWidth=0.5;var sp2=88,nc=~~(W/sp2)+4,nr=~~(H/sp2)+4;for(var i=-nc;i<=nc;i++){ctx.beginPath();ctx.moveTo(i*sp2,-H*2);ctx.lineTo(i*sp2,H*2);ctx.stroke();}for(var j=-nr;j<=nr;j++){ctx.beginPath();ctx.moveTo(-W*2,j*sp2);ctx.lineTo(W*2,j*sp2);ctx.stroke();}ctx.restore();gt+=0.0001;
    [{A:.013,f:.0013,ph:0,c:'#00f5ff'},{A:.009,f:.0017,ph:1.4,c:'#9d4edd'},{A:.011,f:.001,ph:2.7,c:'#ff2d78'}].forEach(function(w){ctx.beginPath();for(var x=0;x<=W;x+=3){var y=H*.5+H*w.A*Math.sin(w.f*x+wt+w.ph);x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.strokeStyle=w.c;ctx.globalAlpha=.045;ctx.lineWidth=1.2;ctx.stroke();});wt+=.005;
    for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){var px=pts[i].x%W,py=pts[i].y%H,qx=pts[j].x%W,qy=pts[j].y%H,dd=Math.sqrt((px-qx)*(px-qx)+(py-qy)*(py-qy));if(dd<115){ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(qx,qy);ctx.strokeStyle=pts[i].c;ctx.globalAlpha=(1-dd/115)*.08;ctx.lineWidth=.55;ctx.stroke();}}
    pts.forEach(function(p){var px=p.x%W,py=p.y%H,dx=px-mx,dy=py-my,dd=Math.sqrt(dx*dx+dy*dy);if(dd<90&&mx>0){p.vx+=dx/dd*0.055;p.vy+=dy/dd*0.055;}p.vx*=0.997;p.vy*=0.997;p.x+=p.vx;p.y+=p.vy;p.lf++;if(p.x<0)p.x+=5000;if(p.x>5000)p.x-=5000;if(p.y<0)p.y+=4000;if(p.y>4000)p.y-=4000;var a=p.a*Math.sin(Math.PI*(p.lf%p.mx)/p.mx);ctx.beginPath();ctx.arc(px,py,p.r,0,6.28);ctx.fillStyle=p.c;ctx.globalAlpha=Math.max(0,a);ctx.fill();});
    bl.forEach(function(b){b.y+=b.vy;if(b.fi){b.a+=.0005;if(b.a>=b.ma)b.fi=false;}else b.a-=.0003;if(b.a<=0||b.y<-80){b.x=Math.random()*W;b.y=H+40;b.a=0;b.fi=true;}ctx.beginPath();var fst=true;for(var i=0;i<=60;i++){var t=(i/60)*b.sc*3-b.sc*1.5,nx=b.x+t,ny=b.y-b.sc*Math.exp(-(t/(b.sc*.9))*(t/(b.sc*.9))/2);if(fst){ctx.moveTo(nx,ny);fst=false;}else ctx.lineTo(nx,ny);}ctx.strokeStyle=b.c;ctx.globalAlpha=b.a;ctx.lineWidth=1.5;ctx.stroke();});
    sy.forEach(function(s){s.y+=s.vy;s.x+=s.dr;if(s.y>H+20){s.y=-20;s.x=Math.random()*W;s.s=SC[~~(Math.random()*SC.length)];}if(s.x<0)s.x=W;if(s.x>W)s.x=0;ctx.font=s.sz+'px monospace';ctx.fillStyle=s.c;ctx.globalAlpha=s.a;ctx.fillText(s.s,s.x,s.y);});
    ob.forEach(function(o){o.x+=o.vx;o.y+=o.vy;o.r+=o.vr;if(o.x<-160)o.x=W+160;if(o.x>W+160)o.x=-160;if(o.y<-160)o.y=H+160;if(o.y>H+160)o.y=-160;var dx=o.x-mx,dy=o.y-my,dd=Math.sqrt(dx*dx+dy*dy);if(dd<200&&mx>0){o.vx+=dx/dd*0.009;o.vy+=dy/dd*0.009;}o.vx*=0.998;o.vy*=0.998;dO(o);});
    tt+=0.016;requestAnimationFrame(loop);
  }
  loop();
})();
</script>"
  style="position:fixed;top:0;left:0;width:0;height:0;border:none;pointer-events:none;z-index:0;"
  sandbox="allow-scripts allow-same-origin">
</iframe>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MATPLOTLIB ULTRA THEME
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "#060d18",
    "axes.facecolor":    "#020408",
    "axes.edgecolor":    "#0a1628",
    "axes.labelcolor":   "#4a6785",
    "text.color":        "#6a8aaa",
    "xtick.color":       "#2a4060",
    "ytick.color":       "#2a4060",
    "grid.color":        "#080f1e",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "font.size":         8.5,
    "figure.dpi":        120,
})

PAGE_COLOR = {
    "landing":       "#00f5ff",
    "lab":           "#00f5ff",
    "tests":         "#ff2d78",
    "ci":            "#ffb700",
    "bayes":         "#00ff9d",
    "combinatorics": "#9d4edd",
}

# ═══════════════════════════════════════════════════════════════
# NAVBAR — ultra glass morphism
# ═══════════════════════════════════════════════════════════════
def navbar():
    page  = st.session_state.page
    color = PAGE_COLOR.get(page, "#00f5ff")
    items = [
        ("landing",       "HOME"),
        ("lab",           "DISTRIB"),
        ("tests",         "HYPOTHESIS"),
        ("bayes",         "BAYES"),
        ("combinatorics", "nCr/nPr"),
        ("ci",            "INTERVALS"),
    ]
    pills = ""
    for pid, lbl in items:
        if pid == page:
            pills += (
                f'<span style="padding:7px 14px;border-radius:6px;font-size:9px;letter-spacing:2.5px;'
                f'color:{color};border:1px solid {color}60;background:linear-gradient(135deg,{color}18,{color}08);'
                f'font-family:Orbitron,monospace;font-weight:700;white-space:nowrap;'
                f'box-shadow:0 0 18px {color}30,inset 0 0 15px {color}08;">{lbl}</span>'
            )
        else:
            pills += (
                f'<span style="padding:7px 14px;border-radius:6px;font-size:9px;letter-spacing:2px;'
                f'color:rgba(100,120,150,0.5);border:1px solid rgba(255,255,255,0.05);'
                f'font-family:Orbitron,monospace;font-weight:500;white-space:nowrap;'
                f'transition:all 0.3s ease;">{lbl}</span>'
            )

    st.markdown(
        f'<div style="position:relative;z-index:100;width:100%;'
        f'background:linear-gradient(180deg,rgba(2,4,8,0.98),rgba(6,13,24,0.95));'
        f'border-bottom:1px solid rgba(255,255,255,0.06);'
        f'backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px);">'
        # animated sweep
        f'<div style="position:absolute;top:0;left:-60%;height:100%;width:50%;'
        f'background:linear-gradient(90deg,transparent,{color}08,transparent);'
        f'animation:shimmer 4s ease-in-out infinite;pointer-events:none;"></div>'
        f'<div style="position:relative;z-index:1;display:flex;align-items:center;'
        f'justify-content:space-between;padding:11px 28px;flex-wrap:wrap;gap:10px;">'
        # logo
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<div style="position:relative;width:28px;height:28px;">'
        f'<div style="position:absolute;inset:0;border-radius:50%;border:1px solid {color}50;'
        f'animation:rotateHalo 4s linear infinite;"></div>'
        f'<div style="position:absolute;inset:4px;border-radius:50%;background:{color};'
        f'opacity:0.9;animation:pulseOrb 2s ease-in-out infinite;"></div>'
        f'</div>'
        f'<span style="font-family:Orbitron,monospace;font-weight:900;font-size:0.85rem;'
        f'background:linear-gradient(90deg,#00f5ff 0%,#ffffff 25%,#9d4edd 50%,#ff2d78 75%,#ffb700 100%);'
        f'background-size:300% 300%;'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:3px;'
        f'animation:shimmer 4s ease-in-out infinite;">'
        f'PROBABILITY LAB</span></div>'
        # pills
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{pills}</div>'
        f'<div style="font-size:8px;color:rgba(255,255,255,0.08);letter-spacing:4px;font-family:monospace;">v3.0</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# UI COMPONENTS — ultra-redesigned
# ═══════════════════════════════════════════════════════════════
def ticker(color="#00f5ff", items=None):
    if items is None:
        items = ["PROBABILITY LAB","MONTE CARLO","HYPOTHESIS TESTING",
                 "BAYES THEOREM","nCr · nPr","CONFIDENCE INTERVALS",
                 "Z-TEST","T-TEST","CHI-SQUARE","10 DISTRIBUTIONS"]
    content = "  ✦  ".join(items * 3)
    st.markdown(
        f'<div style="overflow:hidden;padding:7px 0;position:relative;background:transparent;z-index:10;">'
        f'<div style="position:absolute;inset:0;background:linear-gradient(90deg,rgba(2,4,8,0.95),{color}08,rgba(2,4,8,0.95));border-top:1px solid {color}14;border-bottom:1px solid {color}14;"></div>'
        f'<div style="position:relative;display:inline-block;white-space:nowrap;'
        f'animation:ticker 80s linear infinite;'
        f'font-family:Orbitron,monospace;font-size:9px;color:{color};opacity:.55;letter-spacing:2px;font-weight:500;">'
        f'{content}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{content}</div>'
        f'<div style="position:absolute;left:0;top:0;bottom:0;width:100px;'
        f'background:linear-gradient(90deg,rgba(2,4,8,1),transparent);pointer-events:none;z-index:2;"></div>'
        f'<div style="position:absolute;right:0;top:0;bottom:0;width:100px;'
        f'background:linear-gradient(-90deg,rgba(2,4,8,1),transparent);pointer-events:none;z-index:2;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def section_header(title, sub="", color="#00f5ff", icon=""):
    sub_html = (f'<div style="font-size:11px;color:rgba(200,214,232,0.35);margin-top:4px;'
                f'font-family:JetBrains Mono,monospace;letter-spacing:1px;">{sub}</div>') if sub else ""
    st.markdown(
        f'<div class="plab-page" style="margin:0 0 18px;padding:18px 24px;border-radius:14px;'
        f'background:linear-gradient(135deg,rgba(6,13,24,0.9),rgba(10,22,40,0.7));'
        f'border:1px solid rgba(255,255,255,0.07);'
        f'position:relative;overflow:hidden;backdrop-filter:blur(20px);'
        f'animation:borderFlow 4s ease-in-out infinite;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
        f'background:linear-gradient(90deg,transparent,{color},{color}50,transparent);'
        f'animation:lineGrow 0.8s ease both;"></div>'
        f'<div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;'
        f'border-radius:50%;background:{color};opacity:0.04;filter:blur(30px);"></div>'
        f'<div style="font-size:8px;letter-spacing:5px;color:{color}55;margin-bottom:4px;'
        f'font-family:Orbitron,monospace;font-weight:500;">PROBABILITY LAB</div>'
        f'<div style="font-size:1.4rem;font-weight:900;color:{color};'
        f'text-shadow:0 0 20px {color}60;font-family:Orbitron,monospace;letter-spacing:1px;'
        f'animation:glitchIn 0.6s ease both;">{icon} {title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )

def rcard(label, value, color="#00f5ff", sub=""):
    sub_html = (f'<div style="font-size:9px;color:rgba(200,214,232,0.3);margin-top:3px;font-family:monospace;">{sub}</div>') if sub else ""
    st.markdown(
        f'<div style="padding:13px 15px;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(6,13,24,0.95),rgba(10,22,40,0.8));'
        f'border:1px solid {color}22;margin-bottom:7px;'
        f'position:relative;overflow:hidden;transition:all 0.3s ease;'
        f'animation:cardReveal 0.4s ease both;">'
        f'<div style="position:absolute;left:0;top:0;bottom:0;width:2px;background:{color};opacity:0.6;'
        f'box-shadow:0 0 10px {color};"></div>'
        f'<div style="font-size:8px;color:rgba(200,214,232,0.35);letter-spacing:3px;margin-bottom:4px;'
        f'font-family:Orbitron,monospace;padding-left:8px;">{label}</div>'
        f'<div style="font-size:1.1rem;font-weight:800;color:{color};'
        f'text-shadow:0 0 15px {color}60;font-family:JetBrains Mono,monospace;'
        f'padding-left:8px;animation:numberCount 0.4s ease both;">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )

def verdict(reject, alpha, pv):
    if reject:
        st.markdown(
            f'<div style="padding:14px 16px;border-radius:10px;margin:10px 0;'
            f'background:rgba(255,45,120,0.07);border:1px solid rgba(255,45,120,0.4);'
            f'position:relative;overflow:hidden;">'
            f'<div style="position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,45,120,0.05),transparent);"></div>'
            f'<div style="font-size:10px;color:#ff2d78;letter-spacing:4px;font-weight:800;margin-bottom:5px;'
            f'font-family:Orbitron,monospace;">✗  REJECT H₀</div>'
            f'<div style="font-size:11px;color:rgba(255,150,170,0.9);font-family:JetBrains Mono,monospace;">'
            f'p = {pv:.4f} &lt; α = {alpha} → statistically significant</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="padding:14px 16px;border-radius:10px;margin:10px 0;'
            f'background:rgba(0,255,157,0.06);border:1px solid rgba(0,255,157,0.35);'
            f'position:relative;overflow:hidden;">'
            f'<div style="position:absolute;inset:0;background:linear-gradient(135deg,rgba(0,255,157,0.04),transparent);"></div>'
            f'<div style="font-size:10px;color:#00ff9d;letter-spacing:4px;font-weight:800;margin-bottom:5px;'
            f'font-family:Orbitron,monospace;">✓  FAIL TO REJECT H₀</div>'
            f'<div style="font-size:11px;color:rgba(100,255,180,0.9);font-family:JetBrains Mono,monospace;">'
            f'p = {pv:.4f} ≥ α = {alpha} → not significant</div></div>',
            unsafe_allow_html=True,
        )

def stat_row(items, color="#00f5ff"):
    for col, (lbl, val) in zip(st.columns(len(items)), items):
        with col:
            st.markdown(
                f'<div style="padding:9px 10px;border-radius:8px;'
                f'background:rgba(6,13,24,0.8);border:1px solid {color}18;'
                f'text-align:center;margin-top:8px;position:relative;overflow:hidden;">'
                f'<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
                f'background:linear-gradient(90deg,transparent,{color}40,transparent);"></div>'
                f'<div style="font-size:7px;color:rgba(200,214,232,0.3);letter-spacing:3px;'
                f'font-family:Orbitron,monospace;margin-bottom:3px;">{lbl}</div>'
                f'<div style="font-size:12px;font-weight:800;color:{color};'
                f'font-family:JetBrains Mono,monospace;text-shadow:0 0 10px {color}50;">{val}</div></div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════
# CHARTS — ultra styled
# ═══════════════════════════════════════════════════════════════
def make_chart(samples, tx=None, ty=None, color="#00f5ff", title="",
               discrete=False, bars=True, curve=True,
               extra_lines=None, figsize=(8, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    if samples is not None and bars:
        if discrete:
            v, c = np.unique(samples, return_counts=True)
            ax.bar(v, c / c.sum(), color=color, alpha=0.22, width=0.55, zorder=2,
                   edgecolor=color, linewidth=0.8)
        else:
            n, bins, patches = ax.hist(samples, bins=55, density=True, color=color,
                                       alpha=0.18, zorder=2, edgecolor="none")
            for i, patch in enumerate(patches):
                patch.set_facecolor(color)
                patch.set_alpha(0.10 + 0.20*(i/len(patches)))
    if tx is not None and ty is not None and curve:
        ax.plot(tx, ty, color=color, linewidth=2.5, zorder=3)
        ax.fill_between(tx, ty, alpha=0.05, color=color, zorder=2)
    if extra_lines:
        for ln in extra_lines:
            ax.axvline(ln["x"], color=ln["color"], linewidth=ln.get("lw", 1.5),
                       linestyle=ln.get("ls", "--"), label=ln.get("label", ""), alpha=0.9, zorder=4)
        ax.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa",
                  facecolor="#020408", edgecolor="#0a1628")
    ax.set_title(title, color=color, fontsize=10, fontweight="bold", pad=8, fontfamily="monospace")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    fig.tight_layout(pad=1.2)
    return fig

# ═══════════════════════════════════════════════════════════════
# DISTRIBUTION CONFIG — unchanged logic, just referenced
# ═══════════════════════════════════════════════════════════════
DISTS = {
    "discrete": [
        {"id": "binomial",  "label": "Binomial",  "color": "#00f5ff",
         "params": [("n","Trials n",5,100,1,30),("p","Prob p",0.01,0.99,0.01,0.4),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"np = {p['n']*p['p']:.3f}", f"np(1-p) = {p['n']*p['p']*(1-p['p']):.3f}"),
         "latex": r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}", "is_discrete": True,
         "insight": "Count of successes in n independent Bernoulli trials. Converges to Normal by CLT.",
         "sim":   lambda p, N: simulate_binomial(int(p["n"]), p["p"], N),
         "curve": lambda p: binomial_pmf(int(p["n"]), p["p"])},
        {"id": "geometric", "label": "Geometric", "color": "#9d4edd",
         "params": [("p","Prob p",0.01,0.9,0.01,0.2),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"1/p = {1/p['p']:.3f}", f"(1-p)/p² = {(1-p['p'])/p['p']**2:.3f}"),
         "latex": r"P(X=k)=(1-p)^{k-1}p", "is_discrete": True,
         "insight": "Waiting time until the first success. Has the memoryless property.",
         "sim":   lambda p, N: simulate_geometric(p["p"], N),
         "curve": lambda p: geometric_pmf(p["p"])},
        {"id": "poisson",   "label": "Poisson", "color": "#00ff9d",
         "params": [("lam","Rate λ",0.5,25.0,0.5,5.0),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"λ = {p['lam']}", f"λ = {p['lam']}"),
         "latex": r"P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}", "is_discrete": True,
         "insight": "Count of rare events in a fixed interval. Mean equals variance.",
         "sim":   lambda p, N: simulate_poisson(p["lam"], N),
         "curve": lambda p: poisson_pmf(p["lam"])},
    ],
    "continuous": [
        {"id": "normal",     "label": "Normal", "color": "#ffb700",
         "params": [("mu","Mean μ",-10.0,10.0,0.1,0.0),("sigma","Std σ",0.1,10.0,0.1,1.0),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"μ = {p['mu']}", f"σ² = {p['sigma']**2:.3f}"),
         "latex": r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}", "is_discrete": False,
         "insight": "The bell curve. Symmetric, fully described by μ and σ. CLT convergence target.",
         "sim":   lambda p, N: simulate_normal(p["mu"], p["sigma"], N),
         "curve": lambda p: normal_pdf(p["mu"], p["sigma"])},
        {"id": "exponential","label": "Exponential", "color": "#ff6b35",
         "params": [("lam","Rate λ",0.1,5.0,0.1,1.0),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"1/λ = {1/p['lam']:.3f}", f"1/λ² = {1/p['lam']**2:.3f}"),
         "latex": r"f(x)=\lambda e^{-\lambda x}", "is_discrete": False,
         "insight": "Time between Poisson process events. Continuous memoryless distribution.",
         "sim":   lambda p, N: simulate_exponential(p["lam"], N),
         "curve": lambda p: exponential_pdf(p["lam"])},
        {"id": "gamma",      "label": "Gamma", "color": "#ff2d78",
         "params": [("k","Shape k",0.5,10.0,0.5,2.0),("theta","Scale θ",0.1,5.0,0.1,1.0),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"kθ = {p['k']*p['theta']:.3f}", f"kθ² = {p['k']*p['theta']**2:.3f}"),
         "latex": r"f(x)=\frac{x^{k-1}e^{-x/\theta}}{\theta^k\Gamma(k)}", "is_discrete": False,
         "insight": "Sum of k exponential RVs. Flexible shape for wait times and reliability.",
         "sim":   lambda p, N: simulate_gamma(p["k"], p["theta"], N),
         "curve": lambda p: gamma_pdf(p["k"], p["theta"])},
        {"id": "beta",       "label": "Beta", "color": "#ff8c42",
         "params": [("alpha","Alpha α",0.5,10.0,0.5,2.0),("beta","Beta β",0.5,10.0,0.5,3.0),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (
             f"α/(α+β) = {p['alpha']/(p['alpha']+p['beta']):.4f}",
             f"αβ/((α+β)²(α+β+1)) = {p['alpha']*p['beta']/((p['alpha']+p['beta'])**2*(p['alpha']+p['beta']+1)):.5f}"),
         "latex": r"f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}", "is_discrete": False,
         "insight": "Defined on [0,1]. Models probabilities and proportions. Bayesian conjugate prior.",
         "sim":   lambda p, N: simulate_beta(p["alpha"], p["beta"], N),
         "curve": lambda p: beta_pdf(p["alpha"], p["beta"])},
        {"id": "uniform",    "label": "Uniform", "color": "#00d4aa",
         "params": [("a","Min a",-10.0,0.0,0.5,0.0),("b","Max b",0.1,10.0,0.5,5.0),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"(a+b)/2 = {(p['a']+p['b'])/2:.3f}", f"(b-a)²/12 = {(p['b']-p['a'])**2/12:.4f}"),
         "latex": r"f(x)=\frac{1}{b-a},\;x\in[a,b]", "is_discrete": False,
         "insight": "Every value in [a,b] equally likely. The baseline of randomness.",
         "sim":   lambda p, N: simulate_uniform(p["a"], p["b"], N),
         "curve": lambda p: (lambda x: (x, stats.uniform.pdf(x, p["a"], p["b"]-p["a"])))(
             np.linspace(p["a"]-0.1, p["b"]+0.1, 400))},
        {"id": "chi2",       "label": "Chi-Squared", "icon": "χ²", "color": "#c084fc",
         "params": [("df","Degrees of freedom",1,30,1,5),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (f"df = {int(p['df'])}", f"2·df = {2*int(p['df'])}"),
         "latex": r"f(x)=\frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)}", "is_discrete": False,
         "insight": "Sum of squared standard normals. Used in goodness-of-fit and independence tests.",
         "sim":   lambda p, N: simulate_chi_squared(int(p["df"]), N),
         "curve": lambda p: chi_squared_pdf(int(p["df"]))},
        {"id": "student_t",  "label": "Student-t",   "icon": "t",  "color": "#38bdf8",
         "params": [("df","Degrees of freedom",1,50,1,10),("N","Samples",500,12000,500,5000)],
         "theory": lambda p: (
             "μ = 0  (df > 1)",
             f"df/(df-2) = {int(p['df'])/(int(p['df'])-2):.4f}" if int(p["df"]) > 2 else "σ² = ∞"),
         "latex": r"f(x)\propto\left(1+\frac{x^2}{k}\right)^{-(k+1)/2}", "is_discrete": False,
         "insight": "Normal-like with heavier tails. Used when σ unknown. Approaches Normal as df→∞.",
         "sim":   lambda p, N: simulate_t_distribution(int(p["df"]), N),
         "curve": lambda p: t_distribution_pdf(int(p["df"]))},
    ],
}

# ═══════════════════════════════════════════════════════════════
# PAGE: LANDING — full cinematic redesign
# ═══════════════════════════════════════════════════════════════
def page_landing():
    navbar()
    ticker()

    # ── HERO ─────────────────────────────────────────────────
    st.markdown("""
    <div class="plab-page" style="text-align:center;padding:60px 24px 40px;position:relative;z-index:10;">

      <!-- badge -->
      <div style="display:inline-flex;align-items:center;gap:10px;padding:6px 18px;
           border-radius:30px;border:1px solid rgba(0,245,255,0.15);background:rgba(0,245,255,0.04);
           font-size:8px;letter-spacing:4px;color:rgba(0,245,255,0.5);margin-bottom:28px;
           font-family:Orbitron,monospace;backdrop-filter:blur(10px);">
        <span style="width:5px;height:5px;border-radius:50%;background:#00ff9d;display:inline-block;
              animation:pulseOrb 2s infinite;box-shadow:0 0 8px #00ff9d;"></span>
        LIVE · MONTE CARLO ENGINE · INTERACTIVE STATISTICS
      </div>

      <!-- main title -->
      <div style="position:relative;display:block;margin-bottom:24px;text-align:center;">
        <div style="position:absolute;top:-20px;left:50%;transform:translateX(-50%);
             width:600px;height:160px;background:radial-gradient(ellipse,rgba(0,245,255,0.07) 0%,rgba(157,78,221,0.05) 40%,transparent 70%);
             filter:blur(28px);pointer-events:none;z-index:0;"></div>
        <div style="position:relative;z-index:1;">
          <div style="font-family:Orbitron,monospace;font-size:clamp(2.8rem,8vw,6.5rem);
               font-weight:900;letter-spacing:4px;line-height:1;margin:0;
               background:linear-gradient(135deg,#a8f4ff 0%,#00f5ff 20%,#ffffff 38%,#bf5fff 58%,#ff2d78 78%,#ffb700 100%);
               background-size:300% 300%;
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               animation:shimmer 5s ease infinite,glitchIn 0.9s ease both;
               filter:drop-shadow(0 0 40px rgba(0,245,255,0.22));">PROBABILITY</div>
          <div style="font-family:Orbitron,monospace;font-size:clamp(1.4rem,3.5vw,3rem);
               font-weight:700;letter-spacing:16px;line-height:1;margin-top:4px;
               background:linear-gradient(90deg,#9d4edd,#ff2d78 45%,#ffb700 80%,#00ff9d);
               background-size:200%;
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               animation:shimmer 6s 0.5s ease infinite,glitchIn 0.9s 0.15s ease both;">LAB</div>
        </div>
      </div>

      <p style="font-size:0.85rem;color:rgba(200,214,232,0.35);max-width:380px;margin:0 auto 10px;
           line-height:2;font-family:JetBrains Mono,monospace;letter-spacing:1px;
           animation:fadeIn 1s 0.4s ease both;opacity:0;animation-fill-mode:both;">
        Simulate · Visualize · Test · Understand
      </p>

      <!-- stat counters -->
      <div style="display:flex;justify-content:center;gap:48px;margin-top:24px;flex-wrap:wrap;
           animation:fadeUp 0.8s 0.6s ease both;opacity:0;animation-fill-mode:both;">
        <div style="text-align:center;">
          <div style="font-size:2.4rem;font-weight:900;color:#00f5ff;font-family:Orbitron,monospace;
               text-shadow:0 0 20px rgba(0,245,255,0.5);line-height:1;">10</div>
          <div style="font-size:7px;color:rgba(200,214,232,0.25);letter-spacing:4px;font-family:Orbitron,monospace;margin-top:4px;">DISTRIBUTIONS</div>
        </div>
        <div style="width:1px;background:linear-gradient(180deg,transparent,rgba(255,255,255,0.1),transparent);"></div>
        <div style="text-align:center;">
          <div style="font-size:2.4rem;font-weight:900;color:#9d4edd;font-family:Orbitron,monospace;
               text-shadow:0 0 20px rgba(157,78,221,0.5);line-height:1;">4</div>
          <div style="font-size:7px;color:rgba(200,214,232,0.25);letter-spacing:4px;font-family:Orbitron,monospace;margin-top:4px;">STAT TOOLS</div>
        </div>
        <div style="width:1px;background:linear-gradient(180deg,transparent,rgba(255,255,255,0.1),transparent);"></div>
        <div style="text-align:center;">
          <div style="font-size:2.4rem;font-weight:900;color:#ff2d78;font-family:Orbitron,monospace;
               text-shadow:0 0 20px rgba(255,45,120,0.5);line-height:1;">∞</div>
          <div style="font-size:7px;color:rgba(200,214,232,0.25);letter-spacing:4px;font-family:Orbitron,monospace;margin-top:4px;">SIMULATIONS</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DISTRIBUTION PORTALS ──────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:0 0 20px;position:relative;z-index:10;">
      <div style="font-size:7px;letter-spacing:6px;color:rgba(255,255,255,0.12);
           font-family:Orbitron,monospace;margin-bottom:8px;">CHOOSE YOUR PATH</div>
      <div style="font-size:1.3rem;font-weight:900;
           background:linear-gradient(90deg,#00f5ff,#9d4edd);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-family:Orbitron,monospace;letter-spacing:3px;">ENTER THE LAB</div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 10, 1])
    with mid:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("""
            <a href="?go=disc" style="text-decoration:none;display:block;cursor:pointer;">
            <div class="portal-card scan-wrap" id="card-disc" style="padding:28px;border-radius:16px;position:relative;overflow:hidden;
                 background:linear-gradient(145deg,rgba(0,245,255,0.06),rgba(6,13,24,0.95),rgba(0,245,255,0.02));
                 border:1px solid rgba(0,245,255,0.15);
                 box-shadow:0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(0,245,255,0.1);
                 animation:cardReveal 0.6s 0.2s ease both;opacity:0;animation-fill-mode:both;
                 transition:transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease;">
              <div style="position:absolute;top:0;right:0;width:200px;height:200px;
                   background:radial-gradient(circle,rgba(0,245,255,0.07),transparent 70%);
                   pointer-events:none;"></div>
              <div style="position:absolute;top:0;left:0;right:0;height:1px;
                   background:linear-gradient(90deg,transparent,rgba(0,245,255,0.5),transparent);"></div>
              <div style="font-size:7px;letter-spacing:5px;color:rgba(0,245,255,0.35);
                   margin-bottom:4px;font-family:Orbitron,monospace;">P(X = k)</div>
              <div style="font-size:1.6rem;font-weight:900;color:#00f5ff;
                   font-family:Orbitron,monospace;letter-spacing:2px;
                   text-shadow:0 0 25px rgba(0,245,255,0.5);margin-bottom:2px;">DISCRETE</div>
              <div style="font-size:7px;color:rgba(200,214,232,0.2);letter-spacing:5px;
                   font-family:Orbitron,monospace;margin-bottom:16px;">DISTRIBUTIONS</div>
              <p style="font-size:11px;color:rgba(200,214,232,0.4);font-family:JetBrains Mono,monospace;
                   line-height:1.8;margin:0 0 20px;">Integer-valued outcomes.<br>Counts, successes, arrivals.</p>
              <div style="display:flex;flex-direction:column;gap:9px;margin-bottom:20px;">
                <span style="padding:7px 14px;border-radius:8px;font-size:8.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(0,245,255,0.25);color:#00f5ff;background:rgba(0,245,255,0.07);
                     letter-spacing:1.5px;display:flex;align-items:center;gap:10px;">&nbsp; BINOMIAL</span>
                <span style="padding:7px 14px;border-radius:8px;font-size:8.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(157,78,221,0.25);color:#9d4edd;background:rgba(157,78,221,0.07);
                     letter-spacing:1.5px;display:flex;align-items:center;gap:10px;">&nbsp; GEOMETRIC</span>
                <span style="padding:7px 14px;border-radius:8px;font-size:8.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(0,255,157,0.25);color:#00ff9d;background:rgba(0,255,157,0.07);
                     letter-spacing:1.5px;display:flex;align-items:center;gap:10px;">⚡&nbsp; POISSON</span>
              </div>
              <div style="display:flex;align-items:center;gap:8px;color:rgba(0,245,255,0.4);
                   font-family:Orbitron,monospace;font-size:8.5px;letter-spacing:2px;">
                ENTER MODULE <span style="font-size:13px;">→</span>
              </div>
            </div></a>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <a href="?go=cont" style="text-decoration:none;display:block;cursor:pointer;">
            <div class="portal-card scan-wrap" id="card-cont" style="padding:28px;border-radius:16px;position:relative;overflow:hidden;
                 background:linear-gradient(145deg,rgba(255,183,0,0.06),rgba(6,13,24,0.95),rgba(255,183,0,0.02));
                 border:1px solid rgba(255,183,0,0.15);
                 box-shadow:0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,183,0,0.1);
                 animation:cardReveal 0.6s 0.4s ease both;opacity:0;animation-fill-mode:both;
                 transition:transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease;">
              <div style="position:absolute;top:0;right:0;width:200px;height:200px;
                   background:radial-gradient(circle,rgba(255,183,0,0.07),transparent 70%);
                   pointer-events:none;"></div>
              <div style="position:absolute;top:0;left:0;right:0;height:1px;
                   background:linear-gradient(90deg,transparent,rgba(255,183,0,0.5),transparent);"></div>
              <div style="font-size:7px;letter-spacing:5px;color:rgba(255,183,0,0.35);
                   margin-bottom:4px;font-family:Orbitron,monospace;">f(x) dx</div>
              <div style="font-size:1.6rem;font-weight:900;color:#ffb700;
                   font-family:Orbitron,monospace;letter-spacing:2px;
                   text-shadow:0 0 25px rgba(255,183,0,0.5);margin-bottom:2px;">CONTINUOUS</div>
              <div style="font-size:7px;color:rgba(200,214,232,0.2);letter-spacing:5px;
                   font-family:Orbitron,monospace;margin-bottom:16px;">DISTRIBUTIONS</div>
              <p style="font-size:11px;color:rgba(200,214,232,0.4);font-family:JetBrains Mono,monospace;
                   line-height:1.8;margin:0 0 20px;">Real-valued outcomes.<br>Time, density, proportion.</p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:20px;">
                <span style="padding:7px 12px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;
                     border:1px solid rgba(255,183,0,0.25);color:#ffb700;background:rgba(255,183,0,0.07);
                     letter-spacing:1.2px;white-space:nowrap;">🔔 NORMAL</span>
                <span style="padding:7px 12px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;
                     border:1px solid rgba(255,107,53,0.25);color:#ff6b35;background:rgba(255,107,53,0.07);
                     letter-spacing:1.2px;white-space:nowrap;">📉 EXP</span>
                <span style="padding:7px 12px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;
                     border:1px solid rgba(255,45,120,0.25);color:#ff2d78;background:rgba(255,45,120,0.07);
                     letter-spacing:1.2px;white-space:nowrap;">🌀 GAMMA</span>
                <span style="padding:7px 12px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;
                     border:1px solid rgba(255,140,66,0.25);color:#ff8c42;background:rgba(255,140,66,0.07);
                     letter-spacing:1.2px;white-space:nowrap;">🍊 BETA</span>
                <span style="padding:7px 12px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;
                     border:1px solid rgba(192,132,252,0.25);color:#c084fc;background:rgba(192,132,252,0.07);
                     letter-spacing:1.2px;white-space:nowrap;">χ² CHI-SQ</span>
                <span style="padding:7px 12px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;
                     border:1px solid rgba(56,189,248,0.25);color:#38bdf8;background:rgba(56,189,248,0.07);
                     letter-spacing:1.2px;white-space:nowrap;">t STUDENT</span>
              </div>
              <div style="display:flex;align-items:center;gap:8px;color:rgba(255,183,0,0.4);
                   font-family:Orbitron,monospace;font-size:8.5px;letter-spacing:2px;">
                ENTER MODULE <span style="font-size:13px;">→</span>
              </div>
            </div></a>
            """, unsafe_allow_html=True)

    # ── TOOL TILES ────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:32px 0 20px;position:relative;z-index:10;">
      <div style="font-size:8px;letter-spacing:7px;color:rgba(255,255,255,0.35);
           font-family:Orbitron,monospace;margin-bottom:10px;">ADVANCED MODULES</div>
      <div style="font-size:1.4rem;font-weight:900;color:#ffffff;
           font-family:Orbitron,monospace;letter-spacing:5px;
           text-shadow:0 0 30px rgba(0,245,255,0.4),0 0 60px rgba(157,78,221,0.2);">STATISTICAL TOOLKIT</div>
    </div>
    """, unsafe_allow_html=True)

    _, mid2, _ = st.columns([1, 10, 1])
    with mid2:
        t1, t2, t3, t4 = st.columns(4, gap="medium")
        tools = [
            (t1, "🔬", "HYPOTHESIS TESTING",   "Z · T · Chi-Square tests with p-values", "#ff2d78",  "tests",         "0.8s"),
            (t2, "🧮", "COMBINATORICS",         "nPr · nCr · Multinomial · Birthday",      "#9d4edd",  "comb",          "1.0s"),
            (t3, "🎲", "BAYES THEOREM",         "Prior → Posterior · Beta-Binomial",       "#00ff9d",  "bayes",         "1.2s"),
            (t4, "📏", "CONFIDENCE INTERVALS",  "Z · T · Proportion · Coverage Sim",       "#ffb700",  "ci",            "1.4s"),
        ]
        for col, icon, title_, desc, color, pid, delay in tools:
            with col:
                st.markdown(
                    f'<a href="?go={pid}" style="text-decoration:none;display:block;cursor:pointer;">'
                    f'<div class="tool-tile-{pid} scan-wrap" style="padding:28px 22px;border-radius:16px;position:relative;overflow:hidden;'
                    f'background:linear-gradient(145deg,{color}09,rgba(6,13,24,0.97),{color}04);'
                    f'border:1px solid {color}22;'
                    f'box-shadow:0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 {color}12;'
                    f'animation:cardReveal 0.6s {delay} ease both;opacity:0;animation-fill-mode:both;">'
                    f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
                    f'background:linear-gradient(90deg,transparent,{color}70,transparent);"></div>'
                    f'<div style="position:absolute;top:0;right:0;width:160px;height:160px;'
                    f'border-radius:50%;background:radial-gradient(circle,{color}10,transparent 70%);'
                    f'pointer-events:none;"></div>'
                    f'<div style="font-size:32px;margin-bottom:18px;line-height:1;">{icon}</div>'
                    f'<div style="font-size:11px;font-weight:900;color:{color};letter-spacing:2.5px;'
                    f'font-family:Orbitron,monospace;line-height:1.35;margin-bottom:12px;">{title_}</div>'
                    f'<p style="font-size:9px;color:rgba(200,214,232,0.32);line-height:1.9;margin:0;'
                    f'font-family:JetBrains Mono,monospace;">{desc}</p>'
                    f'</div></a>',
                    unsafe_allow_html=True,
                )

    ticker("#9d4edd", ["Z-TEST","T-TEST","CHI-SQUARE","P-VALUE","ALPHA","BAYES FACTOR",
                       "POSTERIOR","nCr","nPr","CONFIDENCE INTERVAL","TYPE I ERROR","POWER"])

    # ── PROJECT CREDIT ────────────────────────────────────────
    st.markdown(
        '<div style="position:relative;z-index:10;padding:56px 24px 64px;">'
        '<div style="max-width:1100px;margin:0 auto;padding:52px 64px;border-radius:24px;'
        'background:linear-gradient(145deg,rgba(6,13,24,0.98),rgba(10,22,40,0.95),rgba(6,13,24,0.98));'
        'border:1px solid rgba(255,255,255,0.07);'
        'box-shadow:0 30px 90px rgba(0,0,0,0.6),inset 0 1px 0 rgba(255,255,255,0.05);'
        'backdrop-filter:blur(30px);position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:0;left:0;right:0;height:1px;'
        'background:linear-gradient(90deg,transparent,rgba(0,245,255,0.5),rgba(157,78,221,0.5),rgba(255,45,120,0.4),transparent);"></div>'
        '<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
        'background:linear-gradient(90deg,transparent,rgba(255,183,0,0.2),rgba(157,78,221,0.25),transparent);"></div>'
        '<div style="position:absolute;top:-60px;left:-60px;width:280px;height:280px;'
        'border-radius:50%;background:radial-gradient(circle,rgba(0,245,255,0.04),transparent 70%);'
        'filter:blur(30px);pointer-events:none;"></div>'
        '<div style="position:absolute;bottom:-60px;right:-60px;width:280px;height:280px;'
        'border-radius:50%;background:radial-gradient(circle,rgba(157,78,221,0.04),transparent 70%);'
        'filter:blur(30px);pointer-events:none;"></div>'
        '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:40px;">'

        '<div style="flex:1;min-width:260px;">'
        '<div style="font-size:7px;letter-spacing:7px;color:rgba(255,255,255,0.1);'
        'font-family:Orbitron,monospace;margin-bottom:16px;text-transform:uppercase;">PROJECT BY</div>'
        '<div style="font-size:2.2rem;font-weight:900;letter-spacing:3px;line-height:1;'
        'background:linear-gradient(90deg,#00f5ff 0%,#9d4edd 50%,#ff2d78 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'font-family:Orbitron,monospace;margin-bottom:18px;">DESHAN GAUTAM</div>'
        '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        '<span style="padding:5px 14px;border-radius:20px;font-size:8px;letter-spacing:3px;'
        'font-family:Orbitron,monospace;color:rgba(0,245,255,0.6);'
        'border:1px solid rgba(0,245,255,0.15);background:rgba(0,245,255,0.05);">FIRST YEAR</span>'
        '<span style="padding:5px 14px;border-radius:20px;font-size:8px;letter-spacing:2px;'
        'font-family:Orbitron,monospace;color:rgba(157,78,221,0.6);'
        'border:1px solid rgba(157,78,221,0.15);background:rgba(157,78,221,0.05);">IIT MADRAS</span>'
        '<span style="padding:5px 14px;border-radius:20px;font-size:8px;letter-spacing:2px;'
        'font-family:Orbitron,monospace;color:rgba(255,183,0,0.6);'
        'border:1px solid rgba(255,183,0,0.15);background:rgba(255,183,0,0.05);">BS DATA SCIENCE</span>'
        '</div></div>'

        '<div style="width:1px;height:90px;background:linear-gradient(180deg,transparent,rgba(255,255,255,0.08),transparent);flex-shrink:0;"></div>'

        '<div style="display:flex;flex-direction:column;gap:14px;min-width:240px;">'
        '<div style="font-size:7px;letter-spacing:6px;color:rgba(255,255,255,0.1);'
        'font-family:Orbitron,monospace;margin-bottom:4px;">CONNECT</div>'

        '<a href="https://github.com/deshan-5" target="_blank" '
        'style="text-decoration:none;display:flex;align-items:center;gap:14px;padding:13px 18px;'
        'border-radius:12px;background:rgba(6,13,24,0.8);border:1px solid rgba(255,255,255,0.06);'
        'transition:all 0.3s ease;cursor:pointer;" '
        'onmouseover="this.style.borderColor=\'rgba(0,245,255,0.35)\';this.style.background=\'rgba(0,245,255,0.05)\';this.style.boxShadow=\'0 0 20px rgba(0,245,255,0.1)\';" '
        'onmouseout="this.style.borderColor=\'rgba(255,255,255,0.06)\';this.style.background=\'rgba(6,13,24,0.8)\';this.style.boxShadow=\'none\';">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="rgba(200,214,232,0.55)">'
        '<path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.868-.013-1.703-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836a9.59 9.59 0 012.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>'
        '</svg>'
        '<div><div style="font-size:8px;letter-spacing:3px;color:rgba(200,214,232,0.3);'
        'font-family:Orbitron,monospace;margin-bottom:3px;">GITHUB</div>'
        '<div style="font-size:10px;color:rgba(0,245,255,0.7);font-family:JetBrains Mono,monospace;">github.com/deshan-5</div>'
        '</div></a>'

        '<a href="https://www.linkedin.com/in/deshan-gautam-66574331" target="_blank" '
        'style="text-decoration:none;display:flex;align-items:center;gap:14px;padding:13px 18px;'
        'border-radius:12px;background:rgba(6,13,24,0.8);border:1px solid rgba(255,255,255,0.06);'
        'transition:all 0.3s ease;cursor:pointer;" '
        'onmouseover="this.style.borderColor=\'rgba(157,78,221,0.35)\';this.style.background=\'rgba(157,78,221,0.05)\';this.style.boxShadow=\'0 0 20px rgba(157,78,221,0.1)\';" '
        'onmouseout="this.style.borderColor=\'rgba(255,255,255,0.06)\';this.style.background=\'rgba(6,13,24,0.8)\';this.style.boxShadow=\'none\';">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="rgba(200,214,232,0.55)">'
        '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>'
        '</svg>'
        '<div><div style="font-size:8px;letter-spacing:3px;color:rgba(200,214,232,0.3);'
        'font-family:Orbitron,monospace;margin-bottom:3px;">LINKEDIN</div>'
        '<div style="font-size:10px;color:rgba(157,78,221,0.7);font-family:JetBrains Mono,monospace;">deshan-gautam</div>'
        '</div></a>'

        '</div></div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: DISTRIBUTION LAB
# ═══════════════════════════════════════════════════════════════
def page_lab():
    navbar()
    lt    = st.session_state.lab_type
    dists = DISTS[lt]
    ac    = "#00f5ff" if lt == "discrete" else "#ffb700"

    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        # ── elegant back button ──────────────────────────────────
        st.markdown(
            '<a href="?go=landing" style="text-decoration:none;display:inline-flex;align-items:center;'
            'gap:8px;padding:7px 16px;border-radius:8px;margin:16px 0 4px;cursor:pointer;'
            'border:1px solid rgba(255,255,255,0.08);background:rgba(6,13,24,0.7);'
            'font-family:Orbitron,monospace;font-size:8.5px;letter-spacing:2.5px;'
            'color:rgba(200,214,232,0.4);transition:all 0.3s ease;backdrop-filter:blur(10px);"'
            ' onmouseover="this.style.borderColor=\'rgba(0,245,255,0.35)\';this.style.color=\'#00f5ff\';this.style.background=\'rgba(0,245,255,0.06)\';"'
            ' onmouseout="this.style.borderColor=\'rgba(255,255,255,0.08)\';this.style.color=\'rgba(200,214,232,0.4)\';this.style.background=\'rgba(6,13,24,0.7)\';">'
            '<span style="font-size:12px;">←</span> BACK</a>',
            unsafe_allow_html=True,
        )
        section_header(
                "DISCRETE DISTRIBUTIONS" if lt == "discrete" else "CONTINUOUS DISTRIBUTIONS",
                "Integer-valued · PMF" if lt == "discrete" else "Real-valued · PDF",
                ac, "⚀" if lt == "discrete" else "〰",
            )

        ticker(ac)

        tabs = st.tabs([f" {d['label']}" for d in dists])
        for tab, dist in zip(tabs, dists):
            with tab:
                ctrl, chart = st.columns([1, 2.8], gap="large")
                pv = {}

                with ctrl:
                    st.markdown(
                        f'<div style="padding:13px 15px;border-radius:10px;'
                        f'background:linear-gradient(135deg,rgba(6,13,24,0.95),rgba(10,22,40,0.8));'
                        f'border:1px solid {dist["color"]}22;margin-bottom:12px;'
                        f'position:relative;overflow:hidden;">'
                        f'<div style="position:absolute;top:0;left:0;right:0;height:1px;'
                        f'background:linear-gradient(90deg,transparent,{dist["color"]}60,transparent);"></div>'
                        f'<div style="font-size:7px;color:{dist["color"]}55;letter-spacing:4px;margin-bottom:3px;'
                        f'font-family:Orbitron,monospace;">ACTIVE MODULE</div>'
                        f'<div style="font-size:1rem;font-weight:800;color:{dist["color"]};'
                        f'text-shadow:0 0 15px {dist["color"]}50;font-family:Orbitron,monospace;letter-spacing:2px;">'
                        f'{dist["label"].upper()}</div></div>',
                        unsafe_allow_html=True,
                    )
                    for key, lbl, mn, mx, step, default in dist["params"]:
                        if isinstance(default, float):
                            pv[key] = st.slider(lbl, float(mn), float(mx), float(default), float(step), key=f"{dist['id']}_{key}")
                        else:
                            pv[key] = st.slider(lbl, int(mn), int(mx), int(default), int(step), key=f"{dist['id']}_{key}")
                    show_bars  = st.checkbox("Histogram",    True, key=f"{dist['id']}_bars")
                    show_curve = st.checkbox("Theory curve", True, key=f"{dist['id']}_curve")
                    animate    = st.button("▶  ANIMATE", key=f"{dist['id']}_anim", use_container_width=True)
                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                    m, v = dist["theory"](pv)
                    rcard("THEORETICAL MEAN",     m, dist["color"])
                    rcard("THEORETICAL VARIANCE", v, dist["color"])

                with chart:
                    N = int(pv.get("N", 5000))
                    samp    = dist["sim"](pv, N)
                    tx, ty  = dist["curve"](pv)
                    if animate:
                        ph   = st.empty()
                        step = max(80, N // 45)
                        for i in range(step, N + 1, step):
                            fig = make_chart(samp[:i], tx if show_curve else None,
                                             ty if show_curve else None, dist["color"],
                                             f"{dist['label']} (n={i:,})",
                                             dist["is_discrete"], show_bars, show_curve)
                            ph.pyplot(fig, use_container_width=True)
                            plt.close(fig); time.sleep(0.05)
                    else:
                        fig = make_chart(samp, tx if show_curve else None,
                                         ty if show_curve else None, dist["color"],
                                         f"{dist['label']} Distribution",
                                         dist["is_discrete"], show_bars, show_curve)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    stat_row([("MEAN", f"{np.mean(samp):.4f}"), ("STD",  f"{np.std(samp):.4f}"),
                               ("MIN",  f"{np.min(samp):.3f}"),  ("MAX",  f"{np.max(samp):.3f}")],
                             dist["color"])
                    with st.expander("📖  Theory insight"):
                        st.latex(dist["latex"])
                        st.markdown(
                            f'<p style="color:rgba(200,214,232,0.5);font-family:JetBrains Mono,monospace;'
                            f'font-size:11px;line-height:1.9;">{dist["insight"]}</p>',
                            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════
def page_tests():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        st.markdown(
            '<a href="?go=landing" style="text-decoration:none;display:inline-flex;align-items:center;'
            'gap:8px;padding:7px 16px;border-radius:8px;margin:16px 0 4px;cursor:pointer;'
            'border:1px solid rgba(255,255,255,0.08);background:rgba(6,13,24,0.7);'
            'font-family:Orbitron,monospace;font-size:8.5px;letter-spacing:2.5px;'
            'color:rgba(200,214,232,0.4);backdrop-filter:blur(10px);"'
            ' onmouseover="this.style.borderColor=\'rgba(255,45,120,0.45)\';this.style.color=\'#ff2d78\';this.style.background=\'rgba(255,45,120,0.06)\';"'
            ' onmouseout="this.style.borderColor=\'rgba(255,255,255,0.08)\';this.style.color=\'rgba(200,214,232,0.4)\';this.style.background=\'rgba(6,13,24,0.7)\';">'
            '<span style="font-size:12px;">←</span> BACK</a>',
            unsafe_allow_html=True,
        )
        section_header("HYPOTHESIS TESTING", "Z-test · T-test · Chi-Square · Rejection regions", "#ff2d78", "🔬")
        ticker("#ff2d78", ["H₀ NULL HYPOTHESIS","H₁ ALTERNATIVE","P-VALUE","ALPHA LEVEL",
                           "TYPE I ERROR","TYPE II ERROR","CRITICAL VALUE","STATISTICAL POWER"])

        t1, t2, t3 = st.tabs(["⚡  Z / T  One-Sample", "  Two-Sample T", "χ²  Chi-Square"])

        with t1:
            c1, c2 = st.columns([1, 2.6], gap="large")
            with c1:
                ttype = st.selectbox("Test type", ["Z-test (known σ)", "T-test (unknown σ)"], key="ht_type")
                alpha = st.slider("α level", 0.01, 0.20, 0.05, 0.01, key="ht_alpha")
                alt   = st.selectbox("Alternative H₁", ["two-sided", "greater", "less"], key="ht_alt")
                mu0   = st.number_input("Null mean μ₀", value=0.0, key="ht_null")
                st.markdown("<hr style='border-color:rgba(255,255,255,0.05);margin:10px 0;'>", unsafe_allow_html=True)
                if "Z" in ttype:
                    psd = st.number_input("Population σ", value=1.0, min_value=0.01, key="ht_psig")
                    xbr = st.number_input("Sample mean x̄", value=0.5, key="ht_xbar")
                    nn  = st.number_input("n", value=30, min_value=2, key="ht_n")
                    if st.button("RUN Z-TEST  ›", key="rz", use_container_width=True):
                        r = z_test_one_sample(xbr, mu0, psd, nn, alt)
                        rcard("Z-STATISTIC",   f"{r['statistic']:.4f}", "#ff2d78")
                        rcard("P-VALUE",        f"{r['p_value']:.4f}",  "#ff2d78", f"SE = {r['se']:.4f}")
                        cv = get_critical_value("z", alpha, alternative=alt)
                        rcard("CRITICAL VALUE", f"±{cv:.4f}" if alt == "two-sided" else f"{cv:.4f}", "#9d4edd")
                        verdict(r["p_value"] < alpha, alpha, r["p_value"])
                else:
                    if st.checkbox("Generate sample", True, key="ht_gen"):
                        gn = st.slider("n", 10, 200, 40, key="ht_gn")
                        gm = st.number_input("True mean", value=0.3, key="ht_gmu")
                        gs = st.number_input("True std",  value=1.0, min_value=0.01, key="ht_gsd")
                        data = np.random.normal(gm, gs, gn)
                    else:
                        raw = st.text_input("Data (comma-sep)", "2.1,3.4,2.8,4.1,3.9,2.7,3.1,4.0,2.5,3.8", key="ht_raw")
                        try:    data = np.array([float(x.strip()) for x in raw.split(",") if x.strip()])
                        except: data = np.array([0.0])
                    if st.button("RUN T-TEST  ›", key="rt", use_container_width=True) and len(data) > 1:
                        r = t_test_one_sample(data, mu0, alt)
                        rcard("T-STATISTIC",  f"{r['statistic']:.4f}", "#ff2d78")
                        rcard("P-VALUE",       f"{r['p_value']:.4f}",  "#ff2d78", f"df={r['df']}  SE={r['se']:.4f}")
                        rcard("SAMPLE MEAN",   f"{r['sample_mean']:.4f}", "#9d4edd", f"std = {r['sample_std']:.4f}")
                        verdict(r["p_value"] < alpha, alpha, r["p_value"])
            with c2:
                x    = np.linspace(-4.5, 4.5, 500)
                df_p = max(1, int(st.session_state.get("ht_gn", 40)) - 1)
                y    = stats.norm.pdf(x) if "Z" in ttype else stats.t.pdf(x, df_p)
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.grid(True, alpha=0.20, linestyle='--'); ax.plot(x, y, color="#ff2d78", linewidth=2.5)
                ax.fill_between(x, y, alpha=0.06, color="#ff2d78")
                cvt = "z" if "Z" in ttype else "t"; cvdf = None if "Z" in ttype else df_p
                if alt == "two-sided":
                    cv = get_critical_value(cvt, alpha, cvdf)
                    ax.fill_between(x, y, where=x >=  cv, color="#ff2d78", alpha=0.30)
                    ax.fill_between(x, y, where=x <= -cv, color="#ff2d78", alpha=0.30)
                    ax.axvline( cv, color="#ff2d78", linestyle="--", linewidth=1.5, alpha=0.8)
                    ax.axvline(-cv, color="#ff2d78", linestyle="--", linewidth=1.5, alpha=0.8)
                elif alt == "greater":
                    cv = get_critical_value(cvt, alpha, cvdf, "greater")
                    ax.fill_between(x, y, where=x >= cv, color="#ff2d78", alpha=0.30)
                    ax.axvline(cv, color="#ff2d78", linestyle="--", linewidth=1.5, alpha=0.8)
                else:
                    cv = -get_critical_value(cvt, alpha, cvdf, "greater")
                    ax.fill_between(x, y, where=x <= cv, color="#ff2d78", alpha=0.30)
                    ax.axvline(cv, color="#ff2d78", linestyle="--", linewidth=1.5, alpha=0.8)
                ax.set_title("Rejection Region", color="#ff2d78", fontsize=10, fontweight="bold")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

                st.markdown(
                    '<div style="padding:13px 15px;border-radius:10px;'
                    'background:rgba(6,13,24,0.8);border:1px solid rgba(255,255,255,0.06);margin-top:10px;">'
                    '<div style="font-size:7px;color:rgba(200,214,232,0.25);letter-spacing:4px;margin-bottom:8px;'
                    'font-family:Orbitron,monospace;">DECISION RULE</div>'
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;'
                    'font-size:10px;font-family:JetBrains Mono,monospace;">'
                    '<div style="color:#ff2d78;">p &lt; α → Reject H₀</div>'
                    '<div style="color:#00ff9d;">p ≥ α → Fail to Reject</div>'
                    '<div style="color:rgba(200,214,232,0.3);">Type I  = α  (false +)</div>'
                    '<div style="color:rgba(200,214,232,0.3);">Type II = β  (false −)</div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

        with t2:
            ca, cb, cr = st.columns([1, 1, 1.2], gap="large")
            with ca:
                st.markdown('<div style="font-size:8px;color:rgba(200,214,232,0.3);letter-spacing:3px;margin-bottom:4px;font-family:Orbitron,monospace;">GROUP 1</div>', unsafe_allow_html=True)
                if st.checkbox("Generate G1", True, key="t2_g1"):
                    n1 = st.slider("n₁", 5, 200, 30, key="t2_n1")
                    m1 = st.number_input("μ₁", value=5.0, key="t2_m1")
                    s1 = st.number_input("σ₁", value=0.5, min_value=0.01, key="t2_s1")
                    d1 = np.random.normal(m1, s1, n1)
                else:
                    r1 = st.text_area("G1 data", "5.1,4.8,5.3,4.9,5.5,5.0,4.7,5.2", height=65, key="t2_d1")
                    try:    d1 = np.array([float(v.strip()) for v in r1.split(",") if v.strip()])
                    except: d1 = np.array([5.0])
            with cb:
                st.markdown('<div style="font-size:8px;color:rgba(200,214,232,0.3);letter-spacing:3px;margin-bottom:4px;font-family:Orbitron,monospace;">GROUP 2</div>', unsafe_allow_html=True)
                if st.checkbox("Generate G2", True, key="t2_g2"):
                    n2 = st.slider("n₂", 5, 200, 30, key="t2_n2")
                    m2 = st.number_input("μ₂", value=4.7, key="t2_m2")
                    s2 = st.number_input("σ₂", value=0.5, min_value=0.01, key="t2_s2")
                    d2 = np.random.normal(m2, s2, n2)
                else:
                    r2 = st.text_area("G2 data", "4.8,4.5,5.0,4.6,5.1,4.7,4.4,4.9", height=65, key="t2_d2")
                    try:    d2 = np.array([float(v.strip()) for v in r2.split(",") if v.strip()])
                    except: d2 = np.array([4.7])
                al2  = st.slider("α", 0.01, 0.20, 0.05, 0.01, key="t2_a")
                eqv  = st.checkbox("Equal variances", True, key="t2_eq")
                alt2 = st.selectbox("H₁", ["two-sided", "greater", "less"], key="t2_alt")
            with cr:
                if st.button("RUN TWO-SAMPLE T  ›", key="rt2", use_container_width=True) and len(d1) > 1 and len(d2) > 1:
                    res = t_test_two_sample(d1, d2, eqv, alt2)
                    rcard("T-STATISTIC", f"{res['statistic']:.4f}", "#9d4edd")
                    rcard("P-VALUE",      f"{res['p_value']:.4f}",  "#9d4edd")
                    rcard("Δ MEAN",       f"{res['mean1']-res['mean2']:.4f}", "#00ff9d")
                    verdict(res["p_value"] < al2, al2, res["p_value"])
                    fig2, ax2 = plt.subplots(figsize=(5, 2.8))
                    bp = ax2.boxplot([d1, d2], patch_artist=True, widths=0.4,
                                     medianprops=dict(color="#ffffff", linewidth=2))
                    bp["boxes"][0].set_facecolor("#9d4edd18"); bp["boxes"][0].set_edgecolor("#9d4edd60")
                    bp["boxes"][1].set_facecolor("#00ff9d18"); bp["boxes"][1].set_edgecolor("#00ff9d60")
                    ax2.set_xticklabels(["Group 1", "Group 2"])
                    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
                    fig2.tight_layout(pad=1.2); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

        with t3:
            mode = st.radio("Mode", ["Goodness of Fit", "Independence"], horizontal=True, key="chi_mode")
            cx, cy = st.columns([1, 2.2], gap="large")
            with cx:
                ach = st.slider("α", 0.01, 0.20, 0.05, 0.01, key="chi_a")
                if mode == "Goodness of Fit":
                    obs_r = st.text_input("Observed counts", "18,22,25,20,15", key="chi_obs")
                    exp_r = st.text_input("Expected (blank = uniform)", "", key="chi_exp")
                    if st.button("RUN χ² GOF  ›", key="rchi", use_container_width=True):
                        try:
                            ob = [float(v.strip()) for v in obs_r.split(",") if v.strip()]
                            ex = [float(v.strip()) for v in exp_r.split(",") if v.strip()] if exp_r.strip() else None
                            rc = chi_square_gof_test(ob, ex)
                            rcard("χ² STAT", f"{rc['statistic']:.4f}", "#00ff9d")
                            rcard("P-VALUE",  f"{rc['p_value']:.4f}",  "#00ff9d", f"df = {rc['df']}")
                            rcard("CV",       f"{stats.chi2.ppf(1-ach, rc['df']):.4f}", "#9d4edd")
                            verdict(rc["p_value"] < ach, ach, rc["p_value"])
                        except Exception as e: st.error(str(e))
                else:
                    tbl_r = st.text_area("Contingency table", "20,30,10\n15,25,20\n10,20,15", height=75, key="chi_tbl")
                    if st.button("RUN χ² INDEPENDENCE  ›", key="rchi2", use_container_width=True):
                        try:
                            rows = [[float(v.strip()) for v in row.split(",") if v.strip()] for row in tbl_r.strip().split("\n") if row.strip()]
                            ri = chi_square_independence_test(rows)
                            rcard("χ² STAT", f"{ri['statistic']:.4f}", "#00ff9d")
                            rcard("P-VALUE",  f"{ri['p_value']:.4f}",  "#00ff9d", f"df = {ri['df']}")
                            verdict(ri["p_value"] < ach, ach, ri["p_value"])
                        except Exception as e: st.error(str(e))
            with cy:
                df_c = 4; xc = np.linspace(0.01, df_c + 5*np.sqrt(2*df_c), 400)
                fig, ax = plt.subplots(figsize=(8, 3.4))
                ax.plot(xc, stats.chi2.pdf(xc, df_c), color="#00ff9d", linewidth=2.5)
                ax.fill_between(xc, stats.chi2.pdf(xc, df_c), alpha=0.06, color="#00ff9d")
                cvc = stats.chi2.ppf(1 - ach, df_c)
                ax.fill_between(xc, stats.chi2.pdf(xc, df_c), where=xc >= cvc, color="#00ff9d", alpha=0.28)
                ax.axvline(cvc, color="#00ff9d", linestyle="--", linewidth=1.5, alpha=0.8, label=f"CV = {cvc:.3f}")
                ax.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa", facecolor="#020408", edgecolor="#0a1628")
                ax.set_title(f"χ²({df_c}) Distribution", color="#00ff9d", fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.20, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# PAGE: CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════
def page_ci():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        st.markdown(
            '<a href="?go=landing" style="text-decoration:none;display:inline-flex;align-items:center;'
            'gap:8px;padding:7px 16px;border-radius:8px;margin:16px 0 4px;cursor:pointer;'
            'border:1px solid rgba(255,255,255,0.08);background:rgba(6,13,24,0.7);'
            'font-family:Orbitron,monospace;font-size:8.5px;letter-spacing:2.5px;'
            'color:rgba(200,214,232,0.4);backdrop-filter:blur(10px);"'
            ' onmouseover="this.style.borderColor=\'rgba(255,183,0,0.45)\';this.style.color=\'#ffb700\';this.style.background=\'rgba(255,183,0,0.06)\';" '
            ' onmouseout="this.style.borderColor=\'rgba(255,255,255,0.08)\';this.style.color=\'rgba(200,214,232,0.4)\';this.style.background=\'rgba(6,13,24,0.7)\';">'
            '<span style="font-size:12px;">←</span> BACK</a>',
            unsafe_allow_html=True,
        )
        section_header("CONFIDENCE INTERVALS", "Z · T · Proportion · Variance · Coverage Simulation", "#ffb700", "📏")
        ticker("#ffb700", ["95% CI","99% CI","CRITICAL VALUE","STANDARD ERROR","MARGIN OF ERROR","WILSON CI","COVERAGE PROBABILITY"])

        ci1, ci2, ci3, ci4, ci5 = st.tabs([" Mean (Known σ)", " Mean (Unknown σ)", " Proportion", " Variance", " Coverage Sim"])

        with ci1:
            c1, c2 = st.columns([1, 2.5], gap="large")
            with c1:
                xb = st.number_input("Sample mean x̄", value=50.0, key="zi_xb")
                sg = st.number_input("Population σ",   value=10.0, min_value=0.01, key="zi_sg")
                nz = st.number_input("n", value=30, min_value=2, key="zi_n")
                cf = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, key="zi_cf")
                if st.button("COMPUTE Z-INTERVAL  ›", key="rzi", use_container_width=True):
                    r = ci_mean_known_sigma(xb, sg, int(nz), cf)
                    rcard("LOWER", f"{r['lower']:.4f}", "#ffb700")
                    rcard("UPPER", f"{r['upper']:.4f}", "#ffb700")
                    rcard("MARGIN", f"± {r['margin_of_error']:.4f}", "#ff6b35", f"Z* = {r['z_critical']:.4f}")
                    st.success(f"[{r['lower']:.4f},  {r['upper']:.4f}]  at {cf*100:.0f}% confidence")
            with c2:
                sez = sg / np.sqrt(int(nz)); zcv = stats.norm.ppf(1-(1-cf)/2)
                xp  = np.linspace(xb-4*sez, xb+4*sez, 400)
                fig, ax = plt.subplots(figsize=(8, 3.3))
                ax.plot(xp, stats.norm.pdf(xp, xb, sez), color="#ffb700", linewidth=2.5)
                ax.fill_between(xp, stats.norm.pdf(xp, xb, sez), alpha=0.06, color="#ffb700")
                lo, hi = xb-zcv*sez, xb+zcv*sez
                ax.fill_between(xp, stats.norm.pdf(xp, xb, sez), where=(xp>=lo)&(xp<=hi), color="#ffb700", alpha=0.25)
                ax.axvline(lo, color="#ffb700", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.axvline(hi, color="#ffb700", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.set_title(f"Sampling Distribution & {cf*100:.0f}% CI", color="#ffb700", fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.20, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with ci2:
            c1, c2 = st.columns([1, 2.5], gap="large")
            with c1:
                cf_t = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, key="ti_cf")
                if st.checkbox("Generate sample", True, key="ti_gen"):
                    tn  = st.slider("n", 5, 300, 25, key="ti_n")
                    tmu = st.number_input("True mean", value=10.0, key="ti_mu")
                    tsd = st.number_input("True std",  value=3.0, min_value=0.01, key="ti_sd")
                    td  = np.random.normal(tmu, tsd, tn)
                else:
                    raw = st.text_input("Data", "9.1,10.3,8.8,11.2,10.5,9.8,10.1,9.5", key="ti_raw")
                    try:    td = np.array([float(v.strip()) for v in raw.split(",") if v.strip()])
                    except: td = np.array([10.0])
                if st.button("COMPUTE T-INTERVAL  ›", key="rti", use_container_width=True):
                    r = ci_mean_unknown_sigma(td, cf_t)
                    rcard("LOWER", f"{r['lower']:.4f}", "#ff6b35")
                    rcard("UPPER", f"{r['upper']:.4f}", "#ff6b35")
                    rcard("MARGIN", f"± {r['margin_of_error']:.4f}", "#ffb700", f"t* = {r['t_critical']:.4f}  df = {r['df']}")
                    st.info(f"x̄ = {r['sample_mean']:.4f}   SE = {r['se']:.4f}")
            with c2:
                df_t = max(1, len(td)-1); tcv = stats.t.ppf(1-(1-cf_t)/2, df_t)
                xr   = np.linspace(-5, 5, 400)
                fig, ax = plt.subplots(figsize=(8, 3.3))
                ax.plot(xr, stats.t.pdf(xr, df_t), color="#ff6b35", linewidth=2.5, label=f"t({df_t})")
                ax.plot(xr, stats.norm.pdf(xr), color="#334155", linewidth=1, linestyle="--", alpha=0.4, label="N(0,1)")
                ax.fill_between(xr, stats.t.pdf(xr, df_t), alpha=0.06, color="#ff6b35")
                ax.fill_between(xr, stats.t.pdf(xr, df_t), where=(xr>=-tcv)&(xr<=tcv), color="#ff6b35", alpha=0.22)
                ax.axvline(-tcv, color="#ff6b35", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.axvline( tcv, color="#ff6b35", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa", facecolor="#020408", edgecolor="#0a1628")
                ax.set_title(f"t({df_t}) · {cf_t*100:.0f}% CI", color="#ff6b35", fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.20, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with ci3:
            c1, c2 = st.columns([1, 2.5], gap="large")
            with c1:
                sc   = st.number_input("Successes", value=42, min_value=0, key="pi_sc")
                np_  = st.number_input("Total n",   value=100, min_value=1, key="pi_n")
                cf_p = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, key="pi_cf")
                if st.button("COMPUTE WILSON CI  ›", key="rpi", use_container_width=True):
                    r = ci_proportion(int(sc), int(np_), cf_p)
                    rcard("p̂",     f"{r['p_hat']:.4f}",           "#9d4edd")
                    rcard("LOWER", f"{r['lower']:.4f}",           "#9d4edd")
                    rcard("UPPER", f"{r['upper']:.4f}",           "#9d4edd")
                    rcard("MARGIN",f"± {r['margin_of_error']:.4f}","#ffb700", "Wilson method")
            with c2:
                ph  = float(sc) / max(int(np_), 1)
                sep = np.sqrt(ph*(1-ph)/max(int(np_), 1)); zp = stats.norm.ppf(1-(1-cf_p)/2)
                xpp = np.linspace(max(0, ph-0.35), min(1, ph+0.35), 400)
                fig, ax = plt.subplots(figsize=(8, 3.3))
                ax.plot(xpp, stats.norm.pdf(xpp, ph, sep), color="#9d4edd", linewidth=2.5)
                ax.fill_between(xpp, stats.norm.pdf(xpp, ph, sep), alpha=0.06, color="#9d4edd")
                lop, hip = max(0, ph-zp*sep), min(1, ph+zp*sep)
                ax.fill_between(xpp, stats.norm.pdf(xpp, ph, sep), where=(xpp>=lop)&(xpp<=hip), color="#9d4edd", alpha=0.24)
                ax.axvline(lop, color="#9d4edd", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.axvline(hip, color="#9d4edd", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.set_title(f"Proportion CI  p̂ = {ph:.3f}", color="#9d4edd", fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.20, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with ci4:
            c1, c2 = st.columns([1, 2.5], gap="large")
            with c1:
                cf_v = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, key="vi_cf")
                if st.checkbox("Generate data", True, key="vi_gen"):
                    vn  = st.slider("n", 10, 300, 30, key="vi_n")
                    vsg = st.number_input("True σ", value=5.0, min_value=0.1, key="vi_sg")
                    vd  = np.random.normal(0, vsg, vn)
                else:
                    raw = st.text_input("Data", "1.2,3.4,2.1,4.5,2.8,3.3,1.9,4.1", key="vi_raw")
                    try:    vd = np.array([float(v.strip()) for v in raw.split(",") if v.strip()])
                    except: vd = np.array([1.0, 2.0])
                if st.button("COMPUTE VARIANCE CI  ›", key="rvi", use_container_width=True):
                    r = ci_variance(vd, cf_v)
                    rcard("LOWER σ²",  f"{r['lower']:.4f}",          "#ff2d78")
                    rcard("UPPER σ²",  f"{r['upper']:.4f}",          "#ff2d78")
                    rcard("SAMPLE σ²", f"{r['sample_variance']:.4f}", "#ffb700")
            with c2:
                dfv = max(1, len(vd)-1)
                xv  = np.linspace(0.01, dfv + 5*np.sqrt(2*dfv), 400)
                loc = stats.chi2.ppf((1-cf_v)/2, dfv); hic = stats.chi2.ppf(1-(1-cf_v)/2, dfv)
                fig, ax = plt.subplots(figsize=(8, 3.3))
                ax.plot(xv, stats.chi2.pdf(xv, dfv), color="#ff2d78", linewidth=2.5)
                ax.fill_between(xv, stats.chi2.pdf(xv, dfv), alpha=0.06, color="#ff2d78")
                ax.fill_between(xv, stats.chi2.pdf(xv, dfv), where=(xv>=loc)&(xv<=hic), color="#ff2d78", alpha=0.24)
                ax.axvline(loc, color="#ff2d78", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.axvline(hic, color="#ff2d78", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.set_title(f"χ²({dfv}) Variance CI", color="#ff2d78", fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.20, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with ci5:
            c1, c2 = st.columns([1, 2.5], gap="large")
            with c1:
                cmu = st.number_input("True μ", value=50.0, key="cov_mu")
                csg = st.number_input("True σ", value=10.0, key="cov_sg")
                cn  = st.slider("n per CI", 5, 100, 20, key="cov_n")
                ck  = st.slider("# CIs", 20, 200, 60, key="cov_k")
                ccf = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, key="cov_cf")
                run_cov = st.button("SIMULATE COVERAGE  ›", key="rsim", use_container_width=True)
            with c2:
                if run_cov:
                    fig_c, ax_c = plt.subplots(figsize=(8, max(4, int(ck)*0.11)))
                    covered = 0
                    for i in range(int(ck)):
                        s   = np.random.normal(cmu, csg, int(cn))
                        r   = ci_mean_unknown_sigma(s, ccf)
                        hit = r["lower"] <= cmu <= r["upper"]
                        if hit: covered += 1
                        clr = "#00ff9d" if hit else "#ff2d78"
                        ax_c.plot([r["lower"], r["upper"]], [i, i], color=clr, linewidth=0.9, alpha=0.75)
                        ax_c.scatter(r["sample_mean"], i, color=clr, s=8, zorder=3)
                    ax_c.axvline(cmu, color="#ffb700", linewidth=2, linestyle="--", label=f"True μ={cmu}", zorder=4)
                    ax_c.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa", facecolor="#020408", edgecolor="#0a1628")
                    ax_c.set_title(f"Coverage = {covered}/{int(ck)} = {covered/int(ck)*100:.1f}%  (nominal {ccf*100:.0f}%)",
                                   color="#ffb700", fontsize=10, fontweight="bold")
                    ax_c.grid(True, alpha=0.16, axis="x", linestyle='--')
                    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
                    fig_c.tight_layout(pad=1.2); st.pyplot(fig_c, use_container_width=True); plt.close(fig_c)


# ═══════════════════════════════════════════════════════════════
# PAGE: BAYES
# ═══════════════════════════════════════════════════════════════
def page_bayes():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        st.markdown(
            '<a href="?go=landing" style="text-decoration:none;display:inline-flex;align-items:center;'
            'gap:8px;padding:7px 16px;border-radius:8px;margin:16px 0 4px;cursor:pointer;'
            'border:1px solid rgba(255,255,255,0.08);background:rgba(6,13,24,0.7);'
            'font-family:Orbitron,monospace;font-size:8.5px;letter-spacing:2.5px;'
            'color:rgba(200,214,232,0.4);backdrop-filter:blur(10px);"'
            ' onmouseover="this.style.borderColor=\'rgba(0,255,157,0.45)\';this.style.color=\'#00ff9d\';this.style.background=\'rgba(0,255,157,0.06)\';" '
            ' onmouseout="this.style.borderColor=\'rgba(255,255,255,0.08)\';this.style.color=\'rgba(200,214,232,0.4)\';this.style.background=\'rgba(6,13,24,0.7)\';">'
            '<span style="font-size:12px;">←</span> BACK</a>',
            unsafe_allow_html=True,
        )
        section_header("BAYES THEOREM", "Prior · Likelihood · Posterior · Sequential Updating", "#00ff9d", "🎲")
        ticker("#00ff9d", ["P(H|E)=P(E|H)·P(H)/P(E)","PRIOR","POSTERIOR","LIKELIHOOD",
                           "BAYES FACTOR","BETA-BINOMIAL","CONJUGATE PRIOR","SEQUENTIAL UPDATE"])

        b1, b2, b3 = st.tabs([" Classic Bayes", "  Sequential Updates", "  Beta-Binomial Conjugate"])

        with b1:
            c1, c2 = st.columns([1, 2.2], gap="large")
            with c1:
                st.markdown(
                    '<div style="padding:12px 14px;border-radius:10px;'
                    'background:rgba(6,13,24,0.8);border:1px solid rgba(0,255,157,0.1);margin-bottom:12px;">'
                    '<div style="font-size:8px;color:rgba(0,255,157,0.4);letter-spacing:3px;margin-bottom:6px;'
                    'font-family:Orbitron,monospace;">FORMULA</div>'
                    '<div style="font-size:11px;color:#00ff9d;font-family:JetBrains Mono,monospace;line-height:2.2;">'
                    'P(H|E) = P(E|H) · P(H)<br>'
                    '─────────────────────<br>'
                    'P(E|H)·P(H) + P(E|¬H)·P(¬H)</div></div>',
                    unsafe_allow_html=True,
                )
                prior = st.slider("Prior P(H)",           0.01, 0.99, 0.30, 0.01, key="b_prior")
                lt    = st.slider("P(E|H)  Sensitivity",  0.01, 0.99, 0.80, 0.01, key="b_lt")
                lf    = st.slider("P(E|¬H) False +",      0.01, 0.99, 0.10, 0.01, key="b_lf")
                if st.button("COMPUTE POSTERIOR  ›", key="rb", use_container_width=True):
                    r = bayes_theorem(prior, lt, lf)
                    rcard("PRIOR P(H)",       f"{r['prior']:.4f}",     "#475569")
                    rcard("POSTERIOR P(H|E)", f"{r['posterior']:.4f}", "#00ff9d", f"Δ = {r['posterior']-r['prior']:+.4f}")
                    rcard("BAYES FACTOR",     f"{r['bayes_factor']:.3f}", "#9d4edd", "BF > 1 → evidence for H")
            with c2:
                rl = bayes_theorem(prior, lt, lf)
                fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
                for axi, vals, ttl, clr in [
                    (axes[0], [rl["prior"],     1-rl["prior"]],     "Prior",     "#475569"),
                    (axes[1], [rl["posterior"], 1-rl["posterior"]], "Posterior", "#00ff9d"),
                ]:
                    bars = axi.bar(["P(H)", "P(¬H)"], vals, color=["#00ff9d","#ff2d78"], alpha=0.55, width=0.4)
                    for bar in bars: bar.set_edgecolor("none")
                    axi.set_ylim(0, 1); axi.set_title(ttl, color=clr, fontsize=9, fontweight="bold")
                    axi.grid(True, alpha=0.18, axis="y", linestyle='--')
                    axi.spines["top"].set_visible(False); axi.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

                pr    = np.linspace(0.01, 0.99, 200)
                posts = [bayes_theorem(p, lt, lf)["posterior"] for p in pr]
                fig2, ax2 = plt.subplots(figsize=(8, 2.6))
                ax2.plot(pr, posts, color="#00ff9d", linewidth=2.5)
                ax2.fill_between(pr, posts, alpha=0.08, color="#00ff9d")
                ax2.plot(pr, pr, color="#1a2540", linewidth=1, linestyle="--", alpha=0.6, label="no update")
                ax2.axvline(prior, color="#ffb700", linewidth=1.4, linestyle="--", alpha=0.8, label=f"prior={prior}")
                ax2.scatter([prior], [rl["posterior"]], color="#ffb700", s=60, zorder=5)
                ax2.set_xlabel("Prior P(H)"); ax2.set_ylabel("Posterior P(H|E)")
                ax2.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa", facecolor="#020408", edgecolor="#0a1628")
                ax2.grid(True, alpha=0.18, linestyle='--')
                ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
                fig2.tight_layout(pad=1.2); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

        with b2:
            c1, c2 = st.columns([1, 2.2], gap="large")
            with c1:
                p0  = st.slider("Initial prior", 0.01, 0.99, 0.50, 0.01, key="sq_p0")
                nu  = st.slider("Updates", 1, 20, 8, key="sq_n")
                lt2 = st.slider("P(E|H)",  0.10, 0.99, 0.75, 0.01, key="sq_lt")
                lf2 = st.slider("P(E|¬H)", 0.01, 0.90, 0.15, 0.01, key="sq_lf")
                if st.button("SIMULATE UPDATES  ›", key="rsq", use_container_width=True):
                    seq = bayes_update_sequence(p0, [lt2]*int(nu), [lf2]*int(nu))
                    rcard("INITIAL PRIOR",   f"{seq[0]:.4f}",  "#475569")
                    rcard("FINAL POSTERIOR", f"{seq[-1]:.4f}", "#00ff9d", f"after {nu} updates")
            with c2:
                sl  = bayes_update_sequence(p0, [lt2]*int(nu), [lf2]*int(nu))
                fig, ax = plt.subplots(figsize=(8, 3.2))
                ax.plot(range(len(sl)), sl, color="#00ff9d", linewidth=2.5,
                        marker="o", markersize=6, markerfacecolor="#020408",
                        markeredgecolor="#00ff9d", markeredgewidth=2)
                ax.fill_between(range(len(sl)), sl, alpha=0.08, color="#00ff9d")
                ax.axhline(0.5, color="#1a2540", linestyle="--", linewidth=1, alpha=0.6, label="50%")
                ax.set_xlabel("Update number"); ax.set_ylabel("Posterior P(H | evidence)")
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa", facecolor="#020408", edgecolor="#0a1628")
                ax.grid(True, alpha=0.18, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with b3:
            c1, c2 = st.columns([1, 2.2], gap="large")
            with c1:
                ap  = st.slider("Prior α",   0.5, 20.0, 2.0, 0.5, key="bb_a")
                bp  = st.slider("Prior β",   0.5, 20.0, 2.0, 0.5, key="bb_b")
                sc2 = st.slider("Successes", 0, 200, 25, key="bb_s")
                fl2 = st.slider("Failures",  0, 200, 10, key="bb_f")
                if st.button("UPDATE BELIEF  ›", key="rbb", use_container_width=True):
                    r = beta_binomial_update(ap, bp, sc2, fl2)
                    rcard("PRIOR MEAN",     f"{r['prior_mean']:.4f}",     "#475569", f"Beta({ap},{bp})")
                    rcard("POSTERIOR MEAN", f"{r['posterior_mean']:.4f}", "#00ff9d",
                          f"Beta({r['alpha_posterior']:.0f},{r['beta_posterior']:.0f})")
            with c2:
                rbb = beta_binomial_update(ap, bp, sc2, fl2)
                xb2 = np.linspace(0.001, 0.999, 400)
                fig, ax = plt.subplots(figsize=(8, 3.2))
                ax.plot(xb2, stats.beta.pdf(xb2, ap, bp), color="#334155", linewidth=1.8,
                        linestyle="--", label=f"Prior Beta({ap},{bp})", alpha=0.75)
                ax.fill_between(xb2, stats.beta.pdf(xb2, ap, bp), alpha=0.04, color="#334155")
                ax.plot(xb2, stats.beta.pdf(xb2, rbb["alpha_posterior"], rbb["beta_posterior"]),
                        color="#00ff9d", linewidth=2.5,
                        label=f"Posterior Beta({rbb['alpha_posterior']:.0f},{rbb['beta_posterior']:.0f})")
                ax.fill_between(xb2, stats.beta.pdf(xb2, rbb["alpha_posterior"], rbb["beta_posterior"]),
                                alpha=0.10, color="#00ff9d")
                ax.axvline(rbb["prior_mean"],     color="#334155", linewidth=1.2, linestyle=":", alpha=0.8)
                ax.axvline(rbb["posterior_mean"], color="#00ff9d", linewidth=1.5, linestyle=":", alpha=0.9)
                ax.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa", facecolor="#020408", edgecolor="#0a1628")
                ax.set_xlabel("p (probability of success)")
                ax.grid(True, alpha=0.18, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# PAGE: COMBINATORICS
# ═══════════════════════════════════════════════════════════════
def page_combinatorics():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        st.markdown(
            '<a href="?go=landing" style="text-decoration:none;display:inline-flex;align-items:center;'
            'gap:8px;padding:7px 16px;border-radius:8px;margin:16px 0 4px;cursor:pointer;'
            'border:1px solid rgba(255,255,255,0.08);background:rgba(6,13,24,0.7);'
            'font-family:Orbitron,monospace;font-size:8.5px;letter-spacing:2.5px;'
            'color:rgba(200,214,232,0.4);backdrop-filter:blur(10px);"'
            ' onmouseover="this.style.borderColor=\'rgba(157,78,221,0.45)\';this.style.color=\'#9d4edd\';this.style.background=\'rgba(157,78,221,0.06)\';" '
            ' onmouseout="this.style.borderColor=\'rgba(255,255,255,0.08)\';this.style.color=\'rgba(200,214,232,0.4)\';this.style.background=\'rgba(6,13,24,0.7)\';">'
            '<span style="font-size:12px;">←</span> BACK</a>',
            unsafe_allow_html=True,
        )
        section_header("COMBINATORICS", "nPr · nCr · Multinomial · Birthday Problem · List Generator", "#9d4edd", "🧮")
        ticker("#9d4edd", ["nPr = n!/(n-r)!","nCr = n!/r!(n-r)!","MULTINOMIAL",
                           "STARS AND BARS","BIRTHDAY PROBLEM","PASCAL'S TRIANGLE"])

        cc1, cc2, cc3, cc4 = st.tabs([" nPr / nCr", " Multinomial", " Birthday Problem", " List Generator"])

        with cc1:
            c1, c2 = st.columns([1, 1.8], gap="large")
            with c1:
                nc  = st.number_input("n (total)", value=10, min_value=0, max_value=170, key="c_n")
                rc  = st.number_input("r (select)",value=3,  min_value=0, max_value=170, key="c_r")
                rep = st.checkbox("Allow repetition", key="c_rep")
                if st.button("CALCULATE  ›", key="rc_btn", use_container_width=True):
                    perm = int(nc)**int(rc) if rep else permutations(int(nc), int(rc))
                    comb = combinations_with_repetition(int(nc), int(rc)) if rep else combinations(int(nc), int(rc))
                    rcard("PERMUTATIONS nPr", f"{perm:,}", "#9d4edd", "nʳ (rep)" if rep else "n! / (n-r)!")
                    rcard("COMBINATIONS nCr", f"{comb:,}", "#00ff9d", "C(n+r-1,r) (rep)" if rep else "n! / r!(n-r)!")
                    if comb > 0: rcard("RATIO  P/C = r!", f"{perm/comb:,.0f}", "#ffb700")
            with c2:
                nt = min(int(nc), 12)
                fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="#020408")
                ax.set_facecolor("#020408")
                for row in range(nt + 1):
                    for ci in range(row + 1):
                        val   = math.comb(row, ci)
                        xp    = ci - row / 2
                        yp    = nt - row
                        is_t  = (row == min(int(nc), nt)) and (ci == min(int(rc), row))
                        clr   = "#9d4edd" if is_t else "#0a1628"
                        ecl   = "#9d4edd" if is_t else "#1a2540"
                        ax.scatter(xp, yp, s=500, c=clr, zorder=3, edgecolors=ecl, linewidth=1)
                        ax.text(xp, yp, str(val) if val < 1000 else "…",
                                ha="center", va="center", fontsize=6.5,
                                color="#e2e8f0" if is_t else "#334155", fontweight="bold")
                ax.set_title(f"Pascal's Triangle (n ≤ {nt})", color="#9d4edd", fontsize=10, fontweight="bold")
                ax.axis("off"); fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True); plt.close(fig)

        with cc2:
            c1, c2 = st.columns([1, 1.8], gap="large")
            with c1:
                mn = st.number_input("Total n", value=10, min_value=1, key="mn_n")
                mg = st.text_input("Group sizes (sum = n)", "3,3,4", key="mn_g")
                if st.button("COMPUTE  ›", key="rmn", use_container_width=True):
                    try:
                        grps = [int(v.strip()) for v in mg.split(",") if v.strip()]
                        coef = multinomial_coefficient(int(mn), grps)
                        if coef is None:
                            st.error(f"Groups sum to {sum(grps)}, need {mn}")
                        else:
                            rcard("MULTINOMIAL COEFF", f"{coef:,}", "#9d4edd",
                                  f"n! / ({' · '.join(str(g)+'!' for g in grps)})")
                    except Exception as e: st.error(str(e))
            with c2:
                st.markdown(
                    '<div style="padding:16px;border-radius:11px;'
                    'background:rgba(6,13,24,0.8);border:1px solid rgba(157,78,221,0.12);margin-top:6px;">'
                    '<div style="font-size:8px;color:rgba(200,214,232,0.25);letter-spacing:4px;margin-bottom:10px;'
                    'font-family:Orbitron,monospace;">FORMULA</div>'
                    '<div style="font-size:12px;color:#9d4edd;font-family:JetBrains Mono,monospace;line-height:2.3;">'
                    'M = n! / (n₁! · n₂! · ... · nₖ!)<br>'
                    'n₁ + n₂ + ... + nₖ = n</div>'
                    '<div style="font-size:10px;color:rgba(200,214,232,0.3);margin-top:10px;line-height:1.7;'
                    'font-family:JetBrains Mono,monospace;">'
                    'Counts ways to divide n distinct items into k groups of specified sizes.</div></div>',
                    unsafe_allow_html=True,
                )

        with cc3:
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                mp = st.slider("Max people", 10, 100, 60, key="bd_max")
                hl = st.slider("Highlight n =", 2, 100, 23, key="bd_hl")
                pr_val = birthday_problem(int(hl))
                rcard(f"P(shared | n={hl})", f"{pr_val:.4f}", "#9d4edd", f"≈ {pr_val*100:.1f}%")
                if int(hl) >= 23: st.success("n ≥ 23 → probability exceeds 50%! ")
            with c2:
                nr   = range(2, int(mp) + 1)
                prbs = [birthday_problem(n) * 100 for n in nr]
                fig, ax = plt.subplots(figsize=(8, 3.3))
                ax.plot(list(nr), prbs, color="#9d4edd", linewidth=2.5)
                ax.fill_between(list(nr), prbs, alpha=0.10, color="#9d4edd")
                ax.axhline(50, color="#ffb700", linewidth=1.5, linestyle="--", alpha=0.75, label="50%")
                ax.axvline(int(hl), color="#00ff9d", linewidth=1.5, linestyle="--", alpha=0.75, label=f"n={hl}")
                ax.scatter([int(hl)], [birthday_problem(int(hl))*100], color="#00ff9d", s=70, zorder=5)
                ax.set_xlabel("Number of people"); ax.set_ylabel("P(shared birthday) %")
                ax.set_title("The Birthday Problem", color="#9d4edd", fontsize=10, fontweight="bold")
                ax.legend(fontsize=8, framealpha=0.10, labelcolor="#6a8aaa", facecolor="#020408", edgecolor="#0a1628")
                ax.grid(True, alpha=0.18, linestyle='--')
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with cc4:
            c1, c2 = st.columns([1, 1.5], gap="large")
            with c1:
                ir  = st.text_input("Items (comma-sep)", "A,B,C,D,E", key="lg_items")
                rlg = st.number_input("r", value=2, min_value=1, max_value=8, key="lg_r")
                mlg = st.radio("Mode", ["Combinations", "Permutations"], horizontal=True, key="lg_mode")
                if st.button("GENERATE LIST  ›", key="rlg", use_container_width=True):
                    items = [v.strip() for v in ir.split(",") if v.strip()]
                    if len(items) > 8 or int(rlg) > 6:
                        st.warning("Keep n ≤ 8 and r ≤ 6 for listing.")
                    else:
                        lst = (generate_combinations_list(items, int(rlg))
                               if mlg == "Combinations"
                               else generate_permutations_list(items, int(rlg)))
                        rcard("COUNT", f"{len(lst):,}", "#9d4edd")
                        rows_html = "".join(
                            f'<div style="padding:4px 10px;font-size:10px;color:#9d4edd;'
                            f'font-family:JetBrains Mono,monospace;border-bottom:1px solid rgba(157,78,221,0.08);">'
                            f'{"  ·  ".join(r)}</div>'
                            for r in lst[:100]
                        )
                        with c2:
                            st.markdown(
                                f'<div style="font-size:8px;color:rgba(200,214,232,0.25);letter-spacing:3px;margin-bottom:6px;'
                                f'font-family:Orbitron,monospace;">ALL {mlg.upper()}</div>'
                                f'<div style="max-height:300px;overflow-y:auto;background:rgba(6,13,24,0.8);'
                                f'border:1px solid rgba(157,78,221,0.12);border-radius:10px;">{rows_html}</div>',
                                unsafe_allow_html=True,
                            )


# ═══════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════
{
    "landing":       page_landing,
    "lab":           page_lab,
    "tests":         page_tests,
    "ci":            page_ci,
    "bayes":         page_bayes,
    "combinatorics": page_combinatorics,
}.get(st.session_state.page, page_landing)()