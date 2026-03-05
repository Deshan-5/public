import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import streamlit.components.v1 as _stc
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
if "prev_page" not in st.session_state: st.session_state.prev_page = "landing"

def _go(page, lab_type=None):
    st.session_state.prev_page = st.session_state.page
    st.session_state.page = page
    if lab_type: st.session_state.lab_type = lab_type
    st.rerun()


# global css + js particle engine 

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;500;700;900&family=JetBrains+Mono:wght@300;400;600;800&display=swap');

:root {
  --cyan:#00f5ff; --pink:#ff2d78; --amber:#ffb700; --violet:#9d4edd; --green:#00ff9d;
  --bg:#000000; --bg2:#03080f; --bg3:#060d18;
}

*,*::before,*::after{box-sizing:border-box;margin:0}
html,body,[data-testid="stApp"]{background:#000000!important;color:#c8d6e8!important;font-family:'Space Grotesk',sans-serif!important;overflow-x:hidden}
section[data-testid="stSidebar"]{display:none!important}
[data-testid="stHeader"]{display:none!important}
.block-container,[data-testid="stMainBlockContainer"]{padding:0!important;max-width:100%!important}
[data-testid="stVerticalBlock"]>[data-testid="stVerticalBlock"]{gap:0!important}

#plab-canvas{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:0.8}

body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.032) 2px,rgba(0,0,0,0.032) 4px);pointer-events:none;z-index:9999;animation:scanmove 12s linear infinite}
@keyframes scanmove{from{background-position:0 0}to{background-position:0 100vh}}

body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");pointer-events:none;z-index:9998;opacity:0.35}

[data-testid="stSlider"] label{color:rgba(200,214,232,0.4)!important;font-size:10px!important;letter-spacing:3px!important;text-transform:uppercase;font-family:'JetBrains Mono',monospace!important}
[data-testid="stSlider"]>div>div>div{background:linear-gradient(90deg,var(--cyan),var(--violet))!important;height:3px!important;border-radius:2px!important}
[data-testid="stSlider"] [data-testid="stThumbValue"]{background:var(--bg3)!important;border:1px solid rgba(0,245,255,0.4)!important;color:var(--cyan)!important;font-family:'JetBrains Mono',monospace!important;font-size:10px!important}

[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,textarea{background:rgba(6,13,24,0.9)!important;border:1px solid rgba(0,245,255,0.12)!important;color:var(--cyan)!important;border-radius:8px!important;font-family:'JetBrains Mono',monospace!important;font-size:12px!important;transition:all 0.3s ease!important;backdrop-filter:blur(10px)}
[data-testid="stNumberInput"] input:focus,[data-testid="stTextInput"] input:focus,textarea:focus{border-color:rgba(0,245,255,0.5)!important;box-shadow:0 0 0 2px rgba(0,245,255,0.1),inset 0 0 20px rgba(0,245,255,0.04)!important;outline:none}

[data-testid="stSelectbox"]>div{background:rgba(6,13,24,0.9)!important;border:1px solid rgba(0,245,255,0.12)!important;border-radius:8px!important;backdrop-filter:blur(10px)}
[data-testid="stSelectbox"] label,[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{color:rgba(200,214,232,0.5)!important;font-size:10px!important;letter-spacing:2px!important;text-transform:uppercase;font-family:'JetBrains Mono',monospace!important}

[data-testid="stButton"]>button{background:transparent!important;border:1px solid rgba(0,245,255,0.22)!important;color:var(--cyan)!important;font-family:'Orbitron',monospace!important;letter-spacing:3px!important;font-size:9px!important;font-weight:700!important;border-radius:6px!important;padding:10px 18px!important;min-height:40px;transition:all 0.35s cubic-bezier(0.23,1,0.32,1)!important;width:100%;cursor:pointer;position:relative;overflow:hidden;text-transform:uppercase}
[data-testid="stButton"]>button:hover{border-color:rgba(0,245,255,0.65)!important;box-shadow:0 0 20px rgba(0,245,255,0.3),0 0 60px rgba(0,245,255,0.1)!important;transform:translateY(-2px)!important;color:#fff!important}
[data-testid="stButton"]>button:active{transform:translateY(0) scale(0.97)!important}

[data-testid="stTabs"] [role="tablist"]{background:rgba(6,13,24,0.7)!important;border-radius:12px!important;padding:5px!important;border:1px solid rgba(255,255,255,0.05)!important;backdrop-filter:blur(20px)!important;gap:4px!important;margin-bottom:4px!important}
[data-testid="stTabs"] [role="tab"]{color:rgba(200,214,232,0.3)!important;font-family:'JetBrains Mono',monospace!important;font-size:10px!important;letter-spacing:1.5px!important;border-radius:8px!important;min-height:36px!important;padding:0 16px!important;transition:all 0.3s ease!important;cursor:pointer;text-transform:uppercase}
[data-testid="stTabs"] [role="tab"]:hover{color:rgba(200,214,232,0.7)!important}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{background:linear-gradient(135deg,rgba(0,245,255,0.14),rgba(157,78,221,0.09))!important;color:var(--cyan)!important;border:1px solid rgba(0,245,255,0.28)!important;box-shadow:0 0 18px rgba(0,245,255,0.14)!important}
[data-testid="stTabs"] [data-testid="stTabContent"]{padding-top:22px!important}

[data-testid="stExpander"]{background:rgba(6,13,24,0.6)!important;border:1px solid rgba(255,255,255,0.05)!important;border-radius:10px!important;backdrop-filter:blur(10px)}
[data-testid="stExpander"] summary{color:rgba(200,214,232,0.4)!important;font-size:10px!important;letter-spacing:2px!important;font-family:'JetBrains Mono',monospace!important;min-height:40px!important;cursor:pointer}

[data-testid="stSuccess"]{background:rgba(0,255,157,0.06)!important;border:1px solid rgba(0,255,157,0.28)!important;border-radius:10px!important}
[data-testid="stError"]{background:rgba(255,45,120,0.06)!important;border:1px solid rgba(255,45,120,0.28)!important;border-radius:10px!important}
[data-testid="stInfo"]{background:rgba(0,245,255,0.05)!important;border:1px solid rgba(0,245,255,0.22)!important;border-radius:10px!important}
[data-testid="stWarning"]{background:rgba(255,183,0,0.06)!important;border:1px solid rgba(255,183,0,0.28)!important;border-radius:10px!important}

[data-testid="stSlider"]{margin-bottom:14px!important}
[data-testid="stNumberInput"]{margin-bottom:10px!important}
[data-testid="stSelectbox"]{margin-bottom:10px!important}
[data-testid="stCheckbox"]{margin-bottom:8px!important}
[data-testid="stRadio"]{margin-bottom:10px!important}
[data-testid="stTextInput"]{margin-bottom:10px!important}
[data-testid="stTextArea"]{margin-bottom:10px!important}

::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(0,245,255,0.18);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(0,245,255,0.35)}

/* ── KEYFRAMES ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes pulseOrb{0%,100%{opacity:0.75;transform:scale(1)}50%{opacity:1;transform:scale(1.12)}}
@keyframes rotateHalo{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes shimmer{0%{background-position:-200% center}100%{background-position:200% center}}
@keyframes borderFlow{0%{border-color:rgba(0,245,255,0.25);box-shadow:0 0 0 rgba(0,245,255,0)}50%{border-color:rgba(157,78,221,0.45);box-shadow:0 0 20px rgba(157,78,221,0.18)}100%{border-color:rgba(0,245,255,0.25);box-shadow:0 0 0 rgba(0,245,255,0)}}
@keyframes ticker{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes glitchIn{0%{clip-path:inset(0 100% 0 0);opacity:0}40%{clip-path:inset(0 20% 0 0);opacity:0.8}60%{clip-path:inset(0 35% 0 0);opacity:0.6}80%{clip-path:inset(0 5% 0 0);opacity:0.9}100%{clip-path:inset(0 0% 0 0);opacity:1}}
@keyframes cardReveal{from{opacity:0;transform:translateY(28px) scale(0.97);filter:blur(3px)}to{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
@keyframes scanLine{0%{top:-4px;opacity:0.9}100%{top:110%;opacity:0}}
@keyframes lineGrow{from{width:0}to{width:100%}}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
@keyframes numberCount{from{opacity:0;transform:scale(0.5) rotateX(90deg)}to{opacity:1;transform:scale(1) rotateX(0deg)}}
@keyframes aurora{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes borderMarch{0%{background-position:0 0,100% 0,100% 100%,0 100%}100%{background-position:200% 0,100% 200%,-100% 100%,0 -100%}}
@keyframes countUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes glowPulse{0%,100%{text-shadow:0 0 20px rgba(0,245,255,0.5)}50%{text-shadow:0 0 40px rgba(0,245,255,0.9),0 0 80px rgba(0,245,255,0.3)}}
@keyframes tileEntrance{from{opacity:0;transform:translateY(32px) scale(0.95);filter:blur(4px)}to{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
@keyframes hologram{0%{opacity:0.6;transform:scaleY(1)}49%{opacity:0.6;transform:scaleY(1)}50%{opacity:0.3;transform:scaleY(0.98) skewX(0.5deg)}51%{opacity:0.6;transform:scaleY(1)}100%{opacity:0.6;transform:scaleY(1)}}
@keyframes rainbowBorder{0%{border-color:rgba(0,245,255,0.5)}25%{border-color:rgba(157,78,221,0.5)}50%{border-color:rgba(255,45,120,0.5)}75%{border-color:rgba(255,183,0,0.5)}100%{border-color:rgba(0,245,255,0.5)}}
@keyframes neonFlicker{0%,19%,21%,23%,25%,54%,56%,100%{opacity:1}20%,24%,55%{opacity:0.7}}
@keyframes morphBlob{0%,100%{border-radius:60% 40% 30% 70%/60% 30% 70% 40%}50%{border-radius:30% 60% 70% 40%/50% 60% 30% 60%}}
@keyframes scanPulse{0%{box-shadow:0 0 0 0 rgba(0,245,255,0.4)}70%{box-shadow:0 0 0 10px rgba(0,245,255,0)}100%{box-shadow:0 0 0 0 rgba(0,245,255,0)}}

/* ══ PAGE TRANSITIONS ══ */
@keyframes pageEnterFromRight{from{opacity:0;transform:translateX(40px);filter:blur(6px)}to{opacity:1;transform:translateX(0);filter:blur(0)}}
@keyframes pageEnterFromLeft{from{opacity:0;transform:translateX(-40px);filter:blur(6px)}to{opacity:1;transform:translateX(0);filter:blur(0)}}
@keyframes pageEnterUp{from{opacity:0;transform:translateY(30px);filter:blur(4px)}to{opacity:1;transform:translateY(0);filter:blur(0)}}
@keyframes transitionFlash{0%{opacity:0}15%{opacity:1}85%{opacity:1}100%{opacity:0}}
.page-enter{animation:pageEnterFromRight 0.5s cubic-bezier(0.23,1,0.32,1) both}
.page-enter-back{animation:pageEnterFromLeft 0.5s cubic-bezier(0.23,1,0.32,1) both}
.page-enter-up{animation:pageEnterUp 0.45s cubic-bezier(0.23,1,0.32,1) both}

/* ══ STATUS BAR ══ */
#plab-statusbar{
  position:fixed;top:0;left:0;right:0;height:52px;z-index:1000;
  backdrop-filter:blur(24px) saturate(180%);-webkit-backdrop-filter:blur(24px) saturate(180%);
  background:rgba(0,0,0,0.45);
  border-bottom:1px solid rgba(255,255,255,0.06);
  overflow:hidden;
}
#plab-statusbar::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1.5px;
  background:linear-gradient(90deg,#00f5ff,#9d4edd,#ff2d78,#ffb700,#00ff9d,#00f5ff);
  background-size:300% 100%;animation:aurora 3s linear infinite;opacity:0.7;
}
#plab-statusbar::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);
}

/* ══ TICKER — fixed flush under the status bar ══ */
#plab-ticker{
  position:fixed;top:52px;left:0;right:0;height:28px;z-index:999;
  background:rgba(0,0,0,0.5);
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid rgba(0,245,255,0.08);
  overflow:hidden;
}

/* ══ PORTAL CARDS — base styles only, hover handled by JS ══ */
.portal-card{position:relative;border-radius:20px;overflow:hidden;cursor:pointer}
#tile-disc,.portal-card#tile-disc{
  transition:transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease;
  border:1px solid rgba(0,245,255,0.14);
  box-shadow:0 16px 48px rgba(0,0,0,0.45),inset 0 1px 0 rgba(0,245,255,0.07);
  will-change:transform;
}
#tile-cont,.portal-card#tile-cont{
  transition:transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease;
  border:1px solid rgba(255,183,0,0.14);
  box-shadow:0 16px 48px rgba(0,0,0,0.45),inset 0 1px 0 rgba(255,183,0,0.07);
  will-change:transform;
}

/* shimmer sweep on portal cards */
#tile-disc::before,#tile-cont::before{
  content:'';position:absolute;top:0;left:-100%;width:55%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.04),transparent);
  transition:left 0.6s ease;pointer-events:none;z-index:2;
}

/* ══ TOOLKIT TILES — base styles only, hover handled by JS ══ */
[id^="tile-"]{
  transition:transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease!important;
  position:relative;overflow:hidden;cursor:pointer;
  min-height:280px;display:flex;flex-direction:column;
  will-change:transform;
}
[id^="tile-"]::before{content:'';position:absolute;top:0;left:-100%;width:55%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.04),transparent);transition:left 0.55s ease;pointer-events:none;z-index:2}

.tool-tile-tests,.tool-tile-comb,.tool-tile-bayes,.tool-tile-ci{cursor:pointer!important;}

/* glowing text on tile hover */
#tile-disc:hover .tile-title,#tile-cont:hover .tile-title,[id^="tile-"]:hover .tile-title{
  text-shadow:0 0 18px currentColor,0 0 40px currentColor!important;
  transition:text-shadow 0.3s ease;
}

/* perimeter march animation on ALL tiles hover */
[id^="tile-"]::after{
  content:'';position:absolute;inset:0;border-radius:inherit;
  background:linear-gradient(90deg,transparent 20%,rgba(255,255,255,0.08) 50%,transparent 80%);
  background-size:200% 100%;
  opacity:0;transition:opacity 0.3s ease;pointer-events:none;z-index:1;
}
[id^="tile-"]:hover::after{opacity:1;animation:shimmer 1.5s ease infinite;}

/* tile entrance stagger */
#tile-tests{animation:tileEntrance 0.5s 0.1s ease both;opacity:0;animation-fill-mode:both;}
#tile-comb{animation:tileEntrance 0.5s 0.2s ease both;opacity:0;animation-fill-mode:both;}
#tile-bayes{animation:tileEntrance 0.5s 0.3s ease both;opacity:0;animation-fill-mode:both;}
#tile-ci{animation:tileEntrance 0.5s 0.4s ease both;opacity:0;animation-fill-mode:both;}

/* ══ SCAN-WRAP ══ */
.scan-wrap{position:relative;overflow:visible}
.scan-wrap::after{content:'';position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.7),transparent);top:-4px;opacity:0;pointer-events:none;z-index:10}
.scan-wrap:hover::after{animation:scanLine 0.6s ease forwards}

/* ══ MOUSE-TRACK TILE LIT ══ */
.tile-lit{
  background-image:radial-gradient(circle 220px at var(--mx,50%) var(--my,50%),rgba(255,255,255,0.04) 0%,transparent 70%)!important;
}

/* ══ HIDE NAV BUTTONS ══ */
div[data-testid="stButton"]:has(button[data-plab-nav]){
  height:0!important;min-height:0!important;overflow:hidden!important;
  margin:0!important;padding:0!important;opacity:0!important;pointer-events:none!important;
}

/* ══ CRITICAL: let Streamlit markdown containers pass pointer events to tile divs ══ */
[data-testid="stMarkdownContainer"]:has([id^="tile-"]){
  overflow:visible!important;
  pointer-events:auto!important;
}

/* ══ PAGE WRAPPER ══ */
.plab-page{animation:fadeUp 0.5s cubic-bezier(0.23,1,0.32,1) both}
/* ── ENSURE tile hover works — pointer-events fix ── */
div[data-testid="stButton"]:has(button[data-plab-nav]),
div[data-testid="stButton"]:has(button._hidden_nav) {
  pointer-events: none !important;
  height: 0 !important; min-height: 0 !important;
  overflow: hidden !important; margin: 0 !important; padding: 0 !important;
}
/* floatY for social cards */
@keyframes floatY {
  0%,100% { transform: translateY(0px); }
  50%      { transform: translateY(-6px); }
}
/* aurora for footer */
@keyframes aurora {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# ── PAGE TRANSITION FLASH OVERLAY ──
_stc.html("""
<script>
(function(){
  var pd = window.parent.document;

  // CTA hero buttons — scroll to sections
  function setupCTA(){
    var pd = window.parent.document;
    var btnLab  = pd.getElementById('hero-cta-lab');
    var btnDemo = pd.getElementById('hero-cta-demo');
    function scrollTo(id){
      var el = pd.getElementById(id);
      if(el){ var y = el.getBoundingClientRect().top + pd.documentElement.scrollTop - 80; pd.documentElement.scrollTo({top:y,behavior:'smooth'}); }
    }
    if(btnLab  && !btnLab._wired) { btnLab._wired=true;  btnLab.addEventListener('click',  function(){ scrollTo('sec-lab'); }); }
    if(btnDemo && !btnDemo._wired){ btnDemo._wired=true; btnDemo.addEventListener('click', function(){ scrollTo('sec-clt'); }); }
    if(!btnLab || !btnDemo) setTimeout(setupCTA, 400);
  }
  setTimeout(setupCTA, 300);

  // Section scroll helper — forward clicks on navbar anchor links to parent scroll
  function setupNavLinks(){
    pd.querySelectorAll('a[href^="#sec-"]').forEach(function(a){
      if(a._plabScroll) return;
      a._plabScroll = true;
      a.addEventListener('click', function(e){
        e.preventDefault();
        var id = a.getAttribute('href').slice(1);
        var target = pd.getElementById(id);
        if(target){
          var y = target.getBoundingClientRect().top + pd.documentElement.scrollTop - 80;
          pd.documentElement.scrollTo({top: y, behavior: 'smooth'});
        }
      });
    });
    setTimeout(setupNavLinks, 800);
  }
  setTimeout(setupNavLinks, 300);
  // Create overlay if not exists
  var ov = pd.getElementById('plab-transition-overlay');
  if(!ov){
    ov = pd.createElement('div');
    ov.id = 'plab-transition-overlay';
    Object.assign(ov.style,{
      position:'fixed',inset:'0',zIndex:'999990',pointerEvents:'none',
      opacity:'0',transition:'opacity 0.18s ease',
      background:'radial-gradient(ellipse at center, rgba(0,245,255,0.18) 0%, rgba(0,0,0,0.95) 100%)'
    });
    pd.body.appendChild(ov);
  }
  // Flash in then out on page load (signals transition completed)
  ov.style.opacity='1';
  setTimeout(function(){ ov.style.opacity='0'; }, 80);

  // Inject scanline wipe overlay
  var wipe = pd.getElementById('plab-wipe');
  if(!wipe){
    wipe = pd.createElement('div');
    wipe.id = 'plab-wipe';
    Object.assign(wipe.style,{
      position:'fixed',top:'0',left:'0',width:'0',height:'100vh',
      zIndex:'999989',pointerEvents:'none',
      background:'linear-gradient(90deg, transparent, rgba(0,245,255,0.06), rgba(0,245,255,0.03), transparent)',
      transition:'none'
    });
    pd.body.appendChild(wipe);
  }
  // Animate wipe left→right
  wipe.style.transition='none';
  wipe.style.left='-100%';
  wipe.style.width='60%';
  requestAnimationFrame(function(){
    requestAnimationFrame(function(){
      wipe.style.transition='left 0.55s cubic-bezier(0.23,1,0.32,1)';
      wipe.style.left='120%';
      setTimeout(function(){ wipe.style.left='-100%'; wipe.style.transition='none'; }, 600);
    });
  });
})();
</script>
""", height=0, scrolling=False)

# ── ALL 5 VISUAL ENHANCEMENTS ──
_stc.html("""
<script>
(function(){
  var pd = window.parent.document;
  var bod = pd.body;

  // ═══════════════════════════════════════════════
  // 1. MOUSE PARALLAX on floating math symbols
  // ═══════════════════════════════════════════════
  // First inject DOM symbol layer into parent
  if(!pd.getElementById('plab-sym-layer')){
    var symLayer = pd.createElement('div');
    symLayer.id = 'plab-sym-layer';
    Object.assign(symLayer.style,{
      position:'fixed',inset:'0',pointerEvents:'none',zIndex:'2',overflow:'hidden'
    });
    var symList = ['β','μ','σ','π','Σ','λ','∞','Δ','φ','∫','E[X]','P(X)','nCr'];
    var symColors = ['#00f5ff','#9d4edd','#ff2d78','#ffb700','#00ff9d'];
    for(var si=0; si<16; si++){
      var sym = pd.createElement('div');
      sym.className = 'plab-float-sym';
      sym.textContent = symList[si % symList.length];
      sym.setAttribute('data-depth', (0.3 + Math.random()*1.4).toFixed(2));
      Object.assign(sym.style,{
        position:'absolute',
        left: (5 + Math.random()*90) + '%',
        top:  (5 + Math.random()*90) + '%',
        fontFamily:'Orbitron,monospace',
        fontSize: (11 + Math.random()*14) + 'px',
        color: symColors[si % symColors.length],
        opacity: (0.06 + Math.random()*0.1).toFixed(2),
        transition:'transform 0.15s ease',
        userSelect:'none', letterSpacing:'1px', fontWeight:'700'
      });
      symLayer.appendChild(sym);
    }
    bod.appendChild(symLayer);
  }
  function setupParallax(){
    var symbols = pd.querySelectorAll('.plab-float-sym');
    if(!symbols.length){ setTimeout(setupParallax, 800); return; }
    var mx=0, my=0;
    pd.addEventListener('mousemove', function(e){
      mx = (e.clientX / pd.documentElement.clientWidth  - 0.5) * 2;
      my = (e.clientY / pd.documentElement.clientHeight - 0.5) * 2;
    });
    function tick(){
      symbols.forEach(function(s){
        var depth = parseFloat(s.getAttribute('data-depth') || '1');
        s.style.transform = 'translate(' + (mx*depth*18) + 'px,' + (my*depth*12) + 'px)';
      });
      requestAnimationFrame(tick);
    }
    tick();
  }
  setTimeout(setupParallax, 900);

  // ═══════════════════════════════════════════════
  // 2. SCROLL PROGRESS INDICATOR — right edge
  // ═══════════════════════════════════════════════
  if(!pd.getElementById('plab-scroll-track')){
    var track = pd.createElement('div');
    track.id = 'plab-scroll-track';
    Object.assign(track.style,{
      position:'fixed', right:'14px', top:'50%', transform:'translateY(-50%)',
      width:'2px', height:'200px', background:'rgba(255,255,255,0.05)',
      borderRadius:'2px', zIndex:'9990', pointerEvents:'none'
    });
    var fill = pd.createElement('div');
    fill.id = 'plab-scroll-fill';
    Object.assign(fill.style,{
      width:'100%', height:'0%',
      background:'linear-gradient(180deg,#00f5ff,#9d4edd,#ff2d78)',
      borderRadius:'2px', transition:'height 0.12s ease',
      boxShadow:'0 0 6px rgba(0,245,255,0.4)'
    });
    track.appendChild(fill);

    var secs     = ['sec-lab','sec-toolkit','sec-clt','sec-compare','sec-quiz'];
    var secClrs  = ['#00f5ff','#9d4edd','#ff2d78','#ffb700','#00ff9d'];
    var dotRefs  = [];
    secs.forEach(function(id, i){
      var d = pd.createElement('div');
      Object.assign(d.style,{
        position:'absolute', right:'-3px', width:'8px', height:'8px',
        borderRadius:'50%', background:'rgba(255,255,255,0.12)',
        border:'1px solid rgba(255,255,255,0.18)',
        transform:'translateX(50%)', transition:'all 0.28s ease',
        cursor:'pointer', pointerEvents:'all',
        top:(i/(secs.length-1)*100)+'%', marginTop:'-4px'
      });
      d.title = id.replace('sec-','').toUpperCase();
      d.addEventListener('mouseenter', function(){ d.style.transform='translateX(50%) scale(1.5)'; });
      d.addEventListener('mouseleave', function(){ if(!d._active) d.style.transform='translateX(50%) scale(1)'; });
      d.addEventListener('click', function(){
        var el = pd.getElementById(id);
        if(el){ var y = el.getBoundingClientRect().top + pd.documentElement.scrollTop - 80; pd.documentElement.scrollTo({top:y,behavior:'smooth'}); }
      });
      dotRefs.push({el:d, id:id, color:secClrs[i]});
      track.appendChild(d);
    });
    bod.appendChild(track);

    function updateScroll(){
      var s = pd.documentElement.scrollTop;
      var h = pd.documentElement.scrollHeight - pd.documentElement.clientHeight;
      var pct = h > 0 ? Math.min(100, s/h*100) : 0;
      var f = pd.getElementById('plab-scroll-fill');
      if(f) f.style.height = pct + '%';
      dotRefs.forEach(function(dr){
        var el2 = pd.getElementById(dr.id);
        if(!el2) return;
        var elY = el2.getBoundingClientRect().top + s;
        var active = Math.abs(s + 240 - elY) < 340;
        dr._active = active;
        dr.el.style.background  = active ? dr.color : 'rgba(255,255,255,0.12)';
        dr.el.style.boxShadow   = active ? '0 0 8px '+dr.color : 'none';
        dr.el.style.transform   = active ? 'translateX(50%) scale(1.4)' : 'translateX(50%) scale(1)';
      });
    }
    pd.addEventListener('scroll', updateScroll, {passive:true});
    setInterval(updateScroll, 500);
  }

  // ═══════════════════════════════════════════════
  // 3. COUNT-UP ANIMATION — runs when scrolled into view
  // ═══════════════════════════════════════════════
  function setupCountUp(){
    var els = pd.querySelectorAll('[data-countup]');
    if(!els.length){ setTimeout(setupCountUp, 600); return; }
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(entry){
        if(!entry.isIntersecting) return;
        var el = entry.target;
        if(el._counted) return;
        el._counted = true;
        var target = parseInt(el.getAttribute('data-countup'));
        var curr = 0, steps = 40, inc = target/steps, delay = 1000/steps;
        var t = setInterval(function(){
          curr += inc;
          if(curr >= target){ el.textContent = target; clearInterval(t); }
          else { el.textContent = Math.floor(curr); }
        }, delay);
        io.unobserve(el);
      });
    }, {threshold:0.5});
    els.forEach(function(el){ io.observe(el); });
  }
  setTimeout(setupCountUp, 500);

  // ═══════════════════════════════════════════════
  // 4. SECTION REVEAL — fade+slide on scroll enter
  // ═══════════════════════════════════════════════
  if(!pd.getElementById('plab-reveal-css')){
    var revCss = pd.createElement('style');
    revCss.id = 'plab-reveal-css';
    revCss.textContent =
      '.plab-reveal{opacity:0;transform:translateY(40px);filter:blur(5px);' +
      'transition:opacity 0.7s cubic-bezier(0.23,1,0.32,1),' +
      'transform 0.7s cubic-bezier(0.23,1,0.32,1),filter 0.65s ease;}' +
      '.plab-reveal.plab-visible{opacity:1!important;transform:translateY(0)!important;filter:blur(0)!important;}';
    pd.head.appendChild(revCss);
  }
  function setupReveal(){
    var els = pd.querySelectorAll('.plab-reveal');
    if(!els.length){ setTimeout(setupReveal, 700); return; }
    var io2 = new IntersectionObserver(function(entries){
      entries.forEach(function(entry){
        if(!entry.isIntersecting) return;
        var el = entry.target;
        var delay = parseFloat(el.getAttribute('data-reveal-delay')||'0');
        setTimeout(function(){ el.classList.add('plab-visible'); }, delay*1000);
        io2.unobserve(el);
      });
    },{threshold:0.1, rootMargin:'0px 0px -30px 0px'});
    els.forEach(function(el){ io2.observe(el); });
  }
  setTimeout(setupReveal, 600);

  // ═══════════════════════════════════════════════
  // 5. 3D PERSPECTIVE TILT on cards and tiles
  // ═══════════════════════════════════════════════
  function setupTilt(){
    var tiles = pd.querySelectorAll('.portal-card,[id^="tile-"],.scan-wrap');
    if(!tiles.length){ setTimeout(setupTilt, 700); return; }
    tiles.forEach(function(tile){
      if(tile._tilt) return;
      tile._tilt = true;
      // Add shine overlay
      if(!tile.querySelector('.plab-shine')){
        var shine = pd.createElement('div');
        shine.className = 'plab-shine';
        Object.assign(shine.style,{
          position:'absolute',inset:'0',borderRadius:'inherit',
          pointerEvents:'none',zIndex:'5',opacity:'0',
          transition:'opacity 0.2s ease'
        });
        tile.style.position = tile.style.position || 'relative';
        tile.style.overflow = 'hidden';
        tile.appendChild(shine);
      }
      tile.addEventListener('mousemove', function(e){
        var rect = tile.getBoundingClientRect();
        var dx = (e.clientX - rect.left - rect.width/2)  / (rect.width/2);
        var dy = (e.clientY - rect.top  - rect.height/2) / (rect.height/2);
        var gx = ((e.clientX-rect.left)/rect.width)*100;
        var gy = ((e.clientY-rect.top)/rect.height)*100;
        tile.style.transform = 'perspective(900px) rotateX('+(-dy*7)+'deg) rotateY('+(dx*7)+'deg) scale(1.025) translateZ(8px)';
        var sh = tile.querySelector('.plab-shine');
        if(sh){
          sh.style.opacity = '1';
          sh.style.background = 'radial-gradient(circle at '+gx+'% '+gy+'%, rgba(255,255,255,0.07) 0%, transparent 55%)';
        }
      });
      tile.addEventListener('mouseleave', function(){
        tile.style.transform = 'perspective(900px) rotateX(0) rotateY(0) scale(1) translateZ(0)';
        var sh = tile.querySelector('.plab-shine');
        if(sh) sh.style.opacity = '0';
      });
    });
  }
  setTimeout(setupTilt, 700);

})();
</script>
""", height=0, scrolling=False)

# SPACE BG + CURSOR via components iframe
_stc.html("""<script>
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

  // forward tile clicks -> adjacent Streamlit nav buttons
  // ALSO: fix overflow on ALL ancestors so hover transform isn't clipped
  function fwdTiles(){
    var pd=window.parent.document;
    [['tile-disc','_nb_disc'],['tile-cont','_nb_cont'],
     ['tile-tests','_nb_tests'],['tile-comb','_nb_comb'],
     ['tile-bayes','_nb_bayes'],['tile-ci','_nb_ci']
    ].forEach(function(pair){
      var tile=pd.getElementById(pair[0]);
      if(!tile)return;
      // Walk ALL ancestors (not just 8) and force overflow:visible + pointer-events:auto
      var el=tile.parentElement;
      while(el && el!==pd.body){
        el.style.setProperty('overflow','visible','important');
        el.style.setProperty('pointer-events','auto','important');
        el=el.parentElement;
      }
      if(!tile._plab_cf){
        tile._plab_cf=true;
        tile.style.cursor='pointer';
        tile.addEventListener('click',function(e){
          e.stopPropagation();
          var mc2=tile.closest('[data-testid="stMarkdownContainer"]');
          if(!mc2)return;
          var slot=mc2.parentElement;
          var next=slot&&slot.nextElementSibling;
          var btn=next&&next.querySelector('button');
          if(btn){btn.click();}
        });
      }
    });
    setTimeout(fwdTiles,900);
  }
  setTimeout(fwdTiles,500);

  // ── JS-driven hover (bypasses CSS :hover cross-iframe limitation) ──
  var TILE_HOVER = {
    'tile-disc':  {transform:'translateY(-10px) scale(1.02)',  borderColor:'rgba(0,245,255,0.65)',  boxShadow:'0 36px 90px rgba(0,0,0,0.8),0 0 0 1px rgba(0,245,255,0.3),0 0 60px rgba(0,245,255,0.3),inset 0 1px 0 rgba(0,245,255,0.35)'},
    'tile-cont':  {transform:'translateY(-10px) scale(1.02)',  borderColor:'rgba(255,183,0,0.65)',  boxShadow:'0 36px 90px rgba(0,0,0,0.8),0 0 0 1px rgba(255,183,0,0.3),0 0 60px rgba(255,183,0,0.3),inset 0 1px 0 rgba(255,183,0,0.35)'},
    'tile-tests': {transform:'translateY(-10px) scale(1.025)', borderColor:'rgba(255,45,120,0.65)', boxShadow:'0 36px 90px rgba(0,0,0,0.8),0 0 0 1px rgba(255,45,120,0.3),0 0 60px rgba(255,45,120,0.3),inset 0 1px 0 rgba(255,45,120,0.35)'},
    'tile-comb':  {transform:'translateY(-10px) scale(1.025)', borderColor:'rgba(157,78,221,0.65)', boxShadow:'0 36px 90px rgba(0,0,0,0.8),0 0 0 1px rgba(157,78,221,0.3),0 0 60px rgba(157,78,221,0.3),inset 0 1px 0 rgba(157,78,221,0.35)'},
    'tile-bayes': {transform:'translateY(-10px) scale(1.025)', borderColor:'rgba(0,255,157,0.65)',  boxShadow:'0 36px 90px rgba(0,0,0,0.8),0 0 0 1px rgba(0,255,157,0.3),0 0 60px rgba(0,255,157,0.3),inset 0 1px 0 rgba(0,255,157,0.35)'},
    'tile-ci':    {transform:'translateY(-10px) scale(1.025)', borderColor:'rgba(255,183,0,0.65)',  boxShadow:'0 36px 90px rgba(0,0,0,0.8),0 0 0 1px rgba(255,183,0,0.3),0 0 60px rgba(255,183,0,0.3),inset 0 1px 0 rgba(255,183,0,0.35)'},
  };
  var TILE_BASE = {
    'tile-disc':  {borderColor:'rgba(0,245,255,0.14)',  boxShadow:'0 16px 48px rgba(0,0,0,0.45),inset 0 1px 0 rgba(0,245,255,0.07)'},
    'tile-cont':  {borderColor:'rgba(255,183,0,0.14)',  boxShadow:'0 16px 48px rgba(0,0,0,0.45),inset 0 1px 0 rgba(255,183,0,0.07)'},
    'tile-tests': {borderColor:'rgba(255,45,120,0.14)', boxShadow:'0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,45,120,0.07)'},
    'tile-comb':  {borderColor:'rgba(157,78,221,0.14)', boxShadow:'0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(157,78,221,0.07)'},
    'tile-bayes': {borderColor:'rgba(0,255,157,0.14)',  boxShadow:'0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(0,255,157,0.07)'},
    'tile-ci':    {borderColor:'rgba(255,183,0,0.14)',  boxShadow:'0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,183,0,0.07)'},
  };

  function setupTileHover(){
    var pd = window.parent.document;

    // Inject overflow fix CSS (belt)
    var sid = 'plab-overflow-fix';
    if(!pd.getElementById(sid)){
      var s = pd.createElement('style');
      s.id = sid;
      s.textContent = [
        '[data-testid="stColumn"],[data-testid="stVerticalBlock"],',
        '[data-testid="stHorizontalBlock"],[data-testid="stMarkdownContainer"],',
        '[data-testid="column"],[data-testid="stMainBlockContainer"],',
        '.stMainBlockContainer,.main,.block-container{',
        '  overflow:visible!important;',
        '  pointer-events:auto!important;',
        '}',
        '#tile-disc,#tile-cont,#tile-tests,#tile-comb,#tile-bayes,#tile-ci{',
        '  transition:transform 0.4s cubic-bezier(0.23,1,0.32,1),box-shadow 0.4s ease,border-color 0.4s ease!important;',
        '  will-change:transform;cursor:pointer!important;',
        '}'
      ].join('');
      pd.head.appendChild(s);
    }

    // Also walk all ancestors inline (suspenders)
    Object.keys(TILE_HOVER).forEach(function(id){
      var tile = pd.getElementById(id);
      if(!tile) return;
      var el = tile.parentElement;
      while(el && el !== pd.body){
        el.style.setProperty('overflow','visible','important');
        el.style.setProperty('pointer-events','auto','important');
        el = el.parentElement;
      }
    });

    // Use mousemove on parent doc for hit-testing (mouseenter unreliable cross-iframe)
    if(!pd._plabHoverSetup){
      pd._plabHoverSetup = true;
      var hoveredId = null;
      pd.addEventListener('mousemove', function(e){
        var bestId = null;
        Object.keys(TILE_HOVER).forEach(function(id){
          var el = pd.getElementById(id);
          if(!el) return;
          var r = el.getBoundingClientRect();
          if(e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom){
            bestId = id;
          }
        });
        if(bestId !== hoveredId){
          // un-hover old
          if(hoveredId){
            var old = pd.getElementById(hoveredId);
            var base = TILE_BASE[hoveredId];
            if(old && base){
              old.style.setProperty('transform','translateY(0) scale(1)','important');
              old.style.setProperty('border-color', base.borderColor,'important');
              old.style.setProperty('box-shadow', base.boxShadow,'important');
            }
          }
          // hover new
          if(bestId){
            var nel = pd.getElementById(bestId);
            var hov = TILE_HOVER[bestId];
            if(nel && hov){
              nel.style.setProperty('transform', hov.transform,'important');
              nel.style.setProperty('border-color', hov.borderColor,'important');
              nel.style.setProperty('box-shadow', hov.boxShadow,'important');
            }
          }
          hoveredId = bestId;
        }
      });
    }
    // retry until all tiles found
    var allFound = Object.keys(TILE_HOVER).every(function(id){ return !!pd.getElementById(id); });
    if(!allFound) setTimeout(setupTileHover, 400);
  }
  setupTileHover();
  setTimeout(setupTileHover, 600);
  setTimeout(setupTileHover, 1500);
  function runCountUp(){
    var pd=window.parent.document;
    pd.querySelectorAll('[data-countup]').forEach(function(el){
      if(el._counted)return;
      el._counted=true;
      var target=parseInt(el.getAttribute('data-countup'));
      var dur=1200,start=null;
      function step(ts){
        if(!start)start=ts;
        var progress=Math.min((ts-start)/dur,1);
        var ease=1-Math.pow(1-progress,3);
        el.textContent=Math.round(ease*target);
        if(progress<1)requestAnimationFrame(step);
        else el.textContent=target;
      }
      requestAnimationFrame(step);
    });
    setTimeout(runCountUp,800);
  }
  setTimeout(runCountUp,400);

  // Cursor parallax on hero title
  var heroTitle=null;
  function setupParallax(){
    var pd=window.parent.document;
    if(!heroTitle){heroTitle=pd.querySelector('#plab-hero-glow');}
    if(heroTitle){
      pd.addEventListener('mousemove',function(e){
        var cx=pd.documentElement.clientWidth/2;
        var cy=pd.documentElement.clientHeight/2;
        var dx=(e.clientX-cx)/cx;
        var dy=(e.clientY-cy)/cy;
        heroTitle.style.transform='translate(calc(-50% + '+(-dx*18)+'px),'+(-dy*10)+'px)';
      });
    }
    setTimeout(setupParallax,600);
  }
  setTimeout(setupParallax,300);

  loop();
})();
</script>""", height=0, scrolling=False)

# Hide nav buttons and overlay them onto their tiles
_stc.html("""
<script>
(function(){
  function hideNavBtns(){
    var pd=window.parent.document;
    pd.querySelectorAll('button').forEach(function(btn){
      var t=btn.textContent.trim();
      if(t==='enter'){
        // collapse the button wrapper to zero height
        var wrap=btn.closest('[data-testid="stButton"]');
        if(wrap&&!wrap._plab_done){
          wrap._plab_done=true;
          // Find the preceding markdown container (the tile)
          var col=wrap.parentElement;
          var prev=col.previousElementSibling;
          // collapse the wrapper
          wrap.style.cssText='height:0!important;min-height:0!important;overflow:visible!important;margin:0!important;padding:0!important;';
          btn.style.cssText='position:absolute;inset:0;opacity:0;cursor:pointer;border:none;background:transparent;z-index:100;width:100%;height:100%;min-height:0;padding:0;';
          // scroll to top on click
          btn.addEventListener('click', function(){ window.parent.scrollTo({top:0,behavior:'instant'}); });
          // stretch btn over the tile above
          if(prev){
            var h=prev.getBoundingClientRect().height||200;
            btn.style.height=h+'px';
            btn.style.top=(-h)+'px';
          }
        }
      }
    });
    setTimeout(hideNavBtns,600);
  }
  setTimeout(hideNavBtns,200);
})();
</script>
""", height=0, scrolling=False)


# MATPLOTLIB ULTRA THEME

plt.rcParams.update({
    "figure.facecolor":  "#03080f",
    "axes.facecolor":    "#000000",
    "axes.edgecolor":    "#06100a",
    "axes.labelcolor":   "#4a6785",
    "text.color":        "#6a8aaa",
    "xtick.color":       "#2a4060",
    "ytick.color":       "#2a4060",
    "grid.color":        "#060d0a",
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


# minimal status bar — no pills, just identity + breadcrumb

PAGE_LABELS = {
    "landing":       None,
    "lab":           None,
    "tests":         "HYPOTHESIS TESTING",
    "ci":            "CONFIDENCE INTERVALS",
    "bayes":         "BAYES THEOREM",
    "combinatorics": "COMBINATORICS",
}

def _nav_pill(href, label, hc):
    return (
        f'<a href="{href}" '
        f'style="font-family:Orbitron,monospace;font-size:7.5px;font-weight:700;letter-spacing:2px;'
        f'color:rgba(200,214,232,0.55);text-decoration:none;padding:6px 13px;border-radius:20px;'
        f'border:1px solid rgba(255,255,255,0.08);background:transparent;'
        f'transition:all 0.22s ease;cursor:pointer;white-space:nowrap;display:inline-block;" '
        f'onmouseover="this.style.color=\'{hc}\';this.style.borderColor=\'{hc}66\';'
        f'this.style.background=\'{hc}12\';this.style.boxShadow=\'0 0 12px {hc}30\'" '
        f'onmouseout="this.style.color=\'rgba(200,214,232,0.55)\';this.style.borderColor=\'rgba(255,255,255,0.08)\';'
        f'this.style.background=\'transparent\';this.style.boxShadow=\'none\'">'
        f'{label}</a>'
    )

def navbar():
    page  = st.session_state.page
    color = PAGE_COLOR.get(page, "#00f5ff")
    lt    = st.session_state.get("lab_type", "discrete")

    # Build orb + wordmark (shared)
    orb_html = (
        f'<div style="position:relative;width:16px;height:16px;margin-right:10px;flex-shrink:0;">'
        f'<div style="position:absolute;inset:0;border-radius:50%;border:1px solid {color}30;animation:rotateHalo 6s linear infinite;"></div>'
        f'<div style="position:absolute;inset:3px;border-radius:50%;background:{color};opacity:0.85;animation:pulseOrb 3s ease-in-out infinite;"></div>'
        f'</div>'
        f'<span style="font-family:Orbitron,monospace;font-weight:900;font-size:0.72rem;'
        f'color:rgba(255,255,255,0.75);letter-spacing:4px;white-space:nowrap;">PROBABILITY LAB</span>'
    )

    if page == "landing":
        # 5 nav pills centered absolutely
        pills = (
            _nav_pill('#sec-lab',     'LAB',     '#00f5ff') +
            _nav_pill('#sec-toolkit', 'TOOLKIT', '#9d4edd') +
            _nav_pill('#sec-clt',     'CLT',     '#ff2d78') +
            _nav_pill('#sec-compare', 'COMPARE', '#ffb700') +
            _nav_pill('#sec-quiz',    'QUIZ',    '#00ff9d')
        )
        center_html = (
            f'<div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);'
            f'display:flex;align-items:center;gap:3px;">{pills}</div>'
        )
        right_html = (
            f'<div style="margin-left:auto;width:6px;height:6px;border-radius:50%;background:#00ff9d;'
            f'box-shadow:0 0 8px #00ff9d;animation:pulseOrb 2s ease-in-out infinite;flex-shrink:0;"></div>'
        )
        inner = f'<div style="display:flex;align-items:center;">{orb_html}</div>{center_html}{right_html}'
    elif page == "lab":
        sc  = "#00f5ff" if lt == "discrete" else "#ffb700"
        sub = "DISCRETE" if lt == "discrete" else "CONTINUOUS"
        breadcrumb = (
            f'<span style="color:rgba(255,255,255,0.1);margin:0 10px;font-weight:100;">›</span>'
            f'<span style="font-family:Orbitron,monospace;font-size:8px;font-weight:700;color:{sc};letter-spacing:3px;">{sub} DISTRIBUTIONS</span>'
        )
        inner = (
            f'<div style="display:flex;align-items:center;">{orb_html}{breadcrumb}</div>'
            f'<div style="margin-left:auto;width:5px;height:5px;border-radius:50%;background:{color};'
            f'box-shadow:0 0 6px {color};animation:pulseOrb 2s ease-in-out infinite;"></div>'
        )
    else:
        lbl = PAGE_LABELS.get(page, page.upper())
        breadcrumb = (
            f'<span style="color:rgba(255,255,255,0.1);margin:0 10px;font-weight:100;">›</span>'
            f'<span style="font-family:Orbitron,monospace;font-size:8px;font-weight:700;color:{color};letter-spacing:3px;">{lbl}</span>'
        )
        inner = (
            f'<div style="display:flex;align-items:center;">{orb_html}{breadcrumb}</div>'
            f'<div style="margin-left:auto;width:5px;height:5px;border-radius:50%;background:{color};'
            f'box-shadow:0 0 6px {color};animation:pulseOrb 2s ease-in-out infinite;"></div>'
        )

    st.markdown(
        f'<div id="plab-statusbar" style="display:flex;align-items:center;padding:0 24px;position:relative;">'
        f'<div style="position:absolute;bottom:0;left:0;right:0;height:1px;'
        f'background:linear-gradient(90deg,transparent,{color}20,transparent);pointer-events:none;"></div>'
        f'{inner}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:80px;"></div>', unsafe_allow_html=True)

# UI components

def ticker(color="#00f5ff", items=None, attached=False):
    if items is None:
        items = ["PROBABILITY LAB","MONTE CARLO","HYPOTHESIS TESTING",
                 "BAYES THEOREM","nCr · nPr","CONFIDENCE INTERVALS",
                 "Z-TEST","T-TEST","CHI-SQUARE","10 DISTRIBUTIONS"]
    joined = "  ✦  ".join(items)
    double = joined + "  ✦  " + joined
    _stc.html(f"""
<script>
(function(){{
  var pd = window.parent.document;

  // ── Inject CSS for statusbar + ticker into parent doc (ensures fixed positioning works) ──
  if(!pd.getElementById('plab-parent-css')){{
    var css = pd.createElement('style');
    css.id = 'plab-parent-css';
    css.textContent = `
      #plab-statusbar {{
        position:fixed !important;top:0 !important;left:0 !important;right:0 !important;
        height:52px !important;z-index:2147483647 !important;
        backdrop-filter:blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter:blur(24px) saturate(180%) !important;
        background:rgba(2,4,10,0.45) !important;
        border-bottom:none !important;
        overflow:visible !important;
      }}
      #plab-statusbar::after {{
        content:'';position:absolute;bottom:0;left:0;right:0;height:1.5px;
        background:linear-gradient(90deg,#00f5ff,#9d4edd,#ff2d78,#ffb700,#00ff9d,#00f5ff);
        background-size:300% 100%;animation:plab-aurora 3s linear infinite;opacity:0.7;
      }}
      @keyframes plab-aurora {{
        0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}
      }}
      @keyframes plab-pulseOrb {{
        0%,100%{{opacity:0.85;transform:scale(1)}}50%{{opacity:1;transform:scale(1.1)}}
      }}
      @keyframes plab-rotateHalo {{
        from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}
      }}
      #plab-ticker {{
        position:fixed !important;top:52px !important;left:0 !important;right:0 !important;
        height:28px !important;z-index:2147483646 !important;
        background:rgba(2,4,10,0.5) !important;
        backdrop-filter:blur(12px) !important;
        -webkit-backdrop-filter:blur(12px) !important;
        border-bottom:1px solid rgba(0,245,255,0.08) !important;
        overflow:hidden !important;
      }}
      @keyframes plab-ticker-scroll {{
        from{{transform:translateX(0)}}
        to{{transform:translateX(-50%)}}
      }}
    `;
    pd.head.appendChild(css);
  }}

  // ── Build ticker ──
  var t = pd.getElementById('plab-ticker');
  if(!t){{
    t = pd.createElement('div');
    t.id = 'plab-ticker';
    // fade edges
    var fade = pd.createElement('div');
    Object.assign(fade.style, {{
      position:'absolute', inset:'0', zIndex:'2', pointerEvents:'none',
      background:'linear-gradient(90deg, rgba(2,4,10,0.9) 0%, transparent 10%, transparent 90%, rgba(2,4,10,0.9) 100%)'
    }});
    t.appendChild(fade);
    var inner = pd.createElement('div');
    inner.id = 'plab-ticker-inner';
    Object.assign(inner.style, {{
      display:'inline-block', whiteSpace:'nowrap', lineHeight:'28px',
      fontFamily:"'Orbitron',monospace", fontSize:'8px',
      color:'{color}', opacity:'0.55', letterSpacing:'2.5px', fontWeight:'500',
      animation:'plab-ticker-scroll 55s linear infinite',
      willChange:'transform'
    }});
    inner.textContent = '{double}';
    t.appendChild(inner);
    pd.body.appendChild(t);
  }} else {{
    var inner = pd.getElementById('plab-ticker-inner');
    if(inner){{ inner.style.color='{color}'; }}
  }}
}})();
</script>
""", height=0, scrolling=False)
    st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)

def section_divider(color_left="#00f5ff", color_right="#9d4edd", lbl_text="", anchor_id=""):
    """Glowing section separator with optional label and scroll anchor."""
    anchor = f'<div id="{anchor_id}" style="position:relative;top:-90px;"></div>' if anchor_id else ""
    label_html = (
        f'<div style="display:flex;align-items:center;justify-content:center;gap:16px;margin-bottom:0;">'
        f'<div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,{color_left}55);"></div>'
        f'<span style="font-family:Orbitron,monospace;font-size:7px;letter-spacing:5px;'
        f'color:rgba(255,255,255,0.18);white-space:nowrap;padding:0 4px;">{lbl_text}</span>'
        f'<div style="flex:1;height:1px;background:linear-gradient(90deg,{color_right}55,transparent);"></div>'
        f'</div>'
    ) if lbl_text else (
        f'<div style="height:1px;background:linear-gradient(90deg,transparent,{color_left}40,{color_right}40,transparent);"></div>'
    )
    st.markdown(
        f'{anchor}'
        f'<div style="padding:48px 0 8px;position:relative;z-index:10;">'
        f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
        f'width:300px;height:60px;background:radial-gradient(ellipse,{color_left}08 0%,transparent 70%);'
        f'filter:blur(20px);pointer-events:none;"></div>'
        f'{label_html}'
        f'</div>',
        unsafe_allow_html=True
    )


def section_header(title, sub="", color="#00f5ff", icon=""):
    sub_html = (f'<div style="font-size:11px;color:rgba(200,214,232,0.35);margin-top:4px;'
                f'font-family:JetBrains Mono,monospace;letter-spacing:1px;">{sub}</div>') if sub else ""
    st.markdown(
        f'<div class="plab-page page-enter" style="margin:0 0 18px;padding:18px 24px;border-radius:14px;'
        f'background:linear-gradient(135deg,rgba(6,13,24,0.9),rgba(10,22,40,0.7));'
        f'border:1px solid rgba(255,255,255,0.07);'
        f'position:relative;overflow:hidden;backdrop-filter:blur(20px);'
        f'animation:pageEnterFromRight 0.5s cubic-bezier(0.23,1,0.32,1) both, borderFlow 4s 0.5s ease-in-out infinite;">'
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


# CHARTS — ultra styled

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


# DISTRIBUTION CONFIG — unchanged logic, just referenced

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


# PAGE: landing — full cinematic inspired design

def page_landing():
    navbar()
    ticker(attached=True)

    #  Hero section — cinematic fullscreen welcome
    st.markdown("""
    <div class="plab-page page-enter" style="position:relative;z-index:10;min-height:88vh;
         display:flex;flex-direction:column;align-items:center;justify-content:center;
         text-align:center;padding:0 24px 80px;overflow:hidden;">

      <!-- deep radial glow -->
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-60%);
           width:900px;height:500px;pointer-events:none;z-index:0;
           background:radial-gradient(ellipse at center,rgba(0,245,255,0.07) 0%,rgba(157,78,221,0.05) 35%,transparent 70%);
           filter:blur(40px);animation:pulseOrb 6s ease-in-out infinite;"></div>

      <!-- laser beam lines -->
      <div style="position:absolute;inset:0;pointer-events:none;z-index:0;overflow:hidden;">
        <div style="position:absolute;top:0;left:48%;width:2px;height:100%;
             background:linear-gradient(180deg,transparent 0%,rgba(0,245,255,0.12) 20%,rgba(0,245,255,0.35) 50%,rgba(0,245,255,0.12) 80%,transparent 100%);
             filter:blur(1px);animation:hologram 4s ease-in-out infinite;"></div>
        <div style="position:absolute;top:0;left:52%;width:1px;height:100%;
             background:linear-gradient(180deg,transparent 0%,rgba(157,78,221,0.08) 30%,rgba(157,78,221,0.2) 50%,transparent 100%);
             animation:hologram 4s 1s ease-in-out infinite;"></div>
        <div style="position:absolute;top:20%;left:0;width:100%;height:1px;
             background:linear-gradient(90deg,transparent,rgba(0,245,255,0.05),rgba(0,245,255,0.1),rgba(0,245,255,0.05),transparent);"></div>
        <div style="position:absolute;top:75%;left:0;width:100%;height:1px;
             background:linear-gradient(90deg,transparent,rgba(157,78,221,0.06),transparent);"></div>
      </div>

      <!-- badge -->
      <div style="position:relative;z-index:1;display:inline-flex;align-items:center;gap:10px;
           padding:7px 20px;border-radius:30px;border:1px solid rgba(0,245,255,0.18);
           background:rgba(0,245,255,0.05);font-size:8px;letter-spacing:4px;
           color:rgba(0,245,255,0.65);margin-bottom:36px;font-family:Orbitron,monospace;
           backdrop-filter:blur(12px);animation:fadeUp 0.6s 0.1s ease both;opacity:0;animation-fill-mode:both;">
        <span style="width:6px;height:6px;border-radius:50%;background:#00ff9d;display:inline-block;
              animation:pulseOrb 2s infinite;box-shadow:0 0 10px #00ff9d;flex-shrink:0;"></span>
        IIT MADRAS · BS DATA SCIENCE · LIVE MONTE CARLO ENGINE
      </div>

      <!-- WELCOME TO -->
      <div style="position:relative;z-index:1;font-family:Orbitron,monospace;
           font-size:clamp(0.7rem,1.8vw,1rem);font-weight:500;letter-spacing:10px;
           color:rgba(200,214,232,0.4);margin-bottom:10px;
           animation:fadeUp 0.6s 0.2s ease both;opacity:0;animation-fill-mode:both;">
        WELCOME TO
      </div>

      <!-- giant title -->
      <div id="plab-hero-glow" style="position:relative;z-index:1;margin-bottom:8px;
           animation:fadeUp 0.7s 0.3s ease both;opacity:0;animation-fill-mode:both;">
        <div style="font-family:Orbitron,monospace;font-size:clamp(3.5rem,10vw,8rem);
             font-weight:900;letter-spacing:2px;line-height:0.95;
             background:linear-gradient(135deg,#a8f4ff 0%,#00f5ff 25%,#ffffff 45%,#bf5fff 65%,#ff2d78 82%,#ffb700 100%);
             background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
             animation:shimmer 5s ease infinite;">PROBABILITY</div>
        <div style="font-family:Orbitron,monospace;font-size:clamp(2rem,5vw,4.5rem);
             font-weight:900;letter-spacing:18px;line-height:1;margin-top:6px;
             background:linear-gradient(90deg,#9d4edd,#ff2d78 45%,#ffb700 80%,#00ff9d);
             background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
             animation:shimmer 6s 0.5s ease infinite;">LAB</div>
      </div>

      <!-- subtitle -->
      <p style="position:relative;z-index:1;font-size:clamp(0.85rem,1.8vw,1.1rem);
           color:rgba(200,214,232,0.45);max-width:500px;margin:24px auto 0;
           line-height:1.9;font-family:'Space Grotesk',sans-serif;letter-spacing:0.3px;
           animation:fadeUp 0.7s 0.5s ease both;opacity:0;animation-fill-mode:both;">
        Where <em style="color:rgba(0,245,255,0.85);font-style:italic;">curious minds</em> master
        the mathematics of uncertainty.<br>
        Simulate, visualize, and <em style="color:rgba(157,78,221,0.85);font-style:italic;">truly understand</em> probability.
      </p>

      <!-- CTA buttons — use data attributes, JS will wire up scroll -->
      <div style="position:relative;z-index:1;display:flex;gap:16px;justify-content:center;
           margin-top:40px;flex-wrap:wrap;
           animation:fadeUp 0.7s 0.65s ease both;opacity:0;animation-fill-mode:both;">
        <button id="hero-cta-lab"
             style="padding:14px 40px;border-radius:30px;border:none;
             background:linear-gradient(135deg,#00f5ff,#9d4edd);
             font-family:Orbitron,monospace;font-size:9px;font-weight:700;letter-spacing:3px;
             color:#000;cursor:pointer;
             box-shadow:0 0 30px rgba(0,245,255,0.35),0 0 60px rgba(0,245,255,0.12);
             transition:transform 0.25s ease,box-shadow 0.25s ease;"
             onmouseover="this.style.transform='translateY(-3px) scale(1.04)';this.style.boxShadow='0 8px 40px rgba(0,245,255,0.5),0 0 80px rgba(0,245,255,0.18)'"
             onmouseout="this.style.transform='translateY(0) scale(1)';this.style.boxShadow='0 0 30px rgba(0,245,255,0.35),0 0 60px rgba(0,245,255,0.12)'">
          ✦ &nbsp;ENTER THE LAB
        </button>
        <button id="hero-cta-demo"
             style="padding:14px 40px;border-radius:30px;
             border:1px solid rgba(255,255,255,0.18);background:rgba(255,255,255,0.05);
             font-family:Orbitron,monospace;font-size:9px;font-weight:700;letter-spacing:3px;
             color:rgba(200,214,232,0.85);cursor:pointer;backdrop-filter:blur(10px);
             transition:transform 0.25s ease,border-color 0.25s ease,box-shadow 0.25s ease;"
             onmouseover="this.style.transform='translateY(-3px) scale(1.04)';this.style.borderColor='rgba(157,78,221,0.5)';this.style.boxShadow='0 8px 30px rgba(157,78,221,0.25)'"
             onmouseout="this.style.transform='translateY(0) scale(1)';this.style.borderColor='rgba(255,255,255,0.18)';this.style.boxShadow='none'">
          EXPLORE DEMOS
        </button>
      </div>

      <!-- stat counters — 3 only, no quiz -->
      <div style="position:relative;z-index:1;display:flex;justify-content:center;gap:56px;
           margin-top:60px;flex-wrap:wrap;
           animation:fadeUp 0.7s 0.8s ease both;opacity:0;animation-fill-mode:both;">
        <div style="text-align:center;">
          <div data-countup="10" style="font-size:2.4rem;font-weight:900;color:#00f5ff;
               font-family:Orbitron,monospace;text-shadow:0 0 20px rgba(0,245,255,0.5);line-height:1;">10</div>
          <div style="font-size:7px;color:rgba(200,214,232,0.35);letter-spacing:4px;font-family:Orbitron,monospace;margin-top:6px;">DISTRIBUTIONS</div>
        </div>
        <div style="width:1px;background:linear-gradient(180deg,transparent,rgba(255,255,255,0.12),transparent);align-self:stretch;"></div>
        <div style="text-align:center;">
          <div data-countup="4" style="font-size:2.4rem;font-weight:900;color:#9d4edd;
               font-family:Orbitron,monospace;text-shadow:0 0 20px rgba(157,78,221,0.5);line-height:1;">4</div>
          <div style="font-size:7px;color:rgba(200,214,232,0.35);letter-spacing:4px;font-family:Orbitron,monospace;margin-top:6px;">STAT TOOLS</div>
        </div>
        <div style="width:1px;background:linear-gradient(180deg,transparent,rgba(255,255,255,0.12),transparent);align-self:stretch;"></div>
        <div style="text-align:center;">
          <div style="font-size:2.4rem;font-weight:900;color:#ff2d78;
               font-family:Orbitron,monospace;text-shadow:0 0 20px rgba(255,45,120,0.5);line-height:1;">∞</div>
          <div style="font-size:7px;color:rgba(200,214,232,0.35);letter-spacing:4px;font-family:Orbitron,monospace;margin-top:6px;">SIMULATIONS</div>
        </div>
      </div>

      <!-- scroll hint — properly positioned outside flex flow -->
      <div style="position:absolute;bottom:28px;left:0;right:0;
           display:flex;flex-direction:column;align-items:center;gap:8px;
           animation:fadeIn 1s 1.4s ease both;opacity:0;animation-fill-mode:both;z-index:1;">
        <div style="font-size:7px;letter-spacing:5px;color:rgba(255,255,255,0.22);
             font-family:Orbitron,monospace;">SCROLL TO EXPLORE</div>
        <div style="width:1px;height:36px;
             background:linear-gradient(180deg,rgba(0,245,255,0.5),transparent);
             animation:floatY 2s ease-in-out infinite;"></div>
      </div>

    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="text-align:center;padding:0 0 20px;position:relative;z-index:10;" class="plab-reveal">
      <div id="sec-lab" style="position:relative;top:-90px;"></div>
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
            <div id="tile-disc" class="portal-card scan-wrap" style="padding:36px 32px;border-radius:20px;position:relative;overflow:hidden;cursor:pointer;min-height:320px;display:flex;flex-direction:column;justify-content:space-between;background:linear-gradient(145deg,rgba(0,245,255,0.05),rgba(6,13,24,0.97),rgba(0,245,255,0.02));animation:cardReveal 0.6s 0.2s ease both;opacity:0;animation-fill-mode:both;">
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
              <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:24px;">
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(0,245,255,0.2);color:rgba(0,245,255,0.6);
                     letter-spacing:1.5px;">BINOMIAL</span>
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(157,78,221,0.2);color:rgba(157,78,221,0.6);
                     letter-spacing:1.5px;">GEOMETRIC</span>
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(0,255,157,0.2);color:rgba(0,255,157,0.6);
                     letter-spacing:1.5px;">POISSON</span>
              </div>
              <div style="display:flex;align-items:center;justify-content:space-between;">
                <span style="font-family:Orbitron,monospace;font-size:7.5px;letter-spacing:2.5px;
                     color:rgba(0,245,255,0.35);">ENTER MODULE</span>
                <span style="color:rgba(0,245,255,0.4);font-size:18px;line-height:1;">→</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("enter", key="_nb_disc", use_container_width=True): _go("lab","discrete")

        with c2:
            st.markdown("""
            <div id="tile-cont" class="portal-card scan-wrap" style="padding:36px 32px;border-radius:20px;position:relative;overflow:hidden;cursor:pointer;min-height:320px;display:flex;flex-direction:column;justify-content:space-between;background:linear-gradient(145deg,rgba(255,183,0,0.05),rgba(6,13,24,0.97),rgba(255,183,0,0.02));animation:cardReveal 0.6s 0.4s ease both;opacity:0;animation-fill-mode:both;">
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
              <div style="display:flex;gap:7px;flex-wrap:wrap;margin-bottom:24px;">
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(255,183,0,0.2);color:rgba(255,183,0,0.6);
                     letter-spacing:1.2px;">NORMAL</span>
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(255,107,53,0.2);color:rgba(255,107,53,0.6);
                     letter-spacing:1.2px;">EXP</span>
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(255,45,120,0.2);color:rgba(255,45,120,0.6);
                     letter-spacing:1.2px;">GAMMA</span>
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(192,132,252,0.2);color:rgba(192,132,252,0.6);
                     letter-spacing:1.2px;">BETA</span>
                <span style="padding:4px 10px;border-radius:4px;font-size:7.5px;font-family:Orbitron,monospace;
                     border:1px solid rgba(56,189,248,0.2);color:rgba(56,189,248,0.6);
                     letter-spacing:1.2px;">χ² + t</span>
              </div>
              <div style="display:flex;align-items:center;justify-content:space-between;">
                <span style="font-family:Orbitron,monospace;font-size:7.5px;letter-spacing:2.5px;
                     color:rgba(255,183,0,0.35);">ENTER MODULE</span>
                <span style="color:rgba(255,183,0,0.4);font-size:18px;line-height:1;">→</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("enter", key="_nb_cont", use_container_width=True): _go("lab","continuous")

    # ── Statistical tools section.
    section_divider("#00f5ff", "#9d4edd", "· · ·", anchor_id="sec-toolkit")
    st.markdown("""
    <div style="text-align:center;padding:8px 0 20px;position:relative;z-index:10;" class="plab-reveal" data-reveal-delay="0.1">
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
                    f'<div id="tile-{pid}" class="tool-tile-{pid} scan-wrap" style="padding:28px 22px 22px;border-radius:16px;position:relative;overflow:hidden;cursor:pointer;'
                    f'background:linear-gradient(145deg,{color}08,rgba(6,13,24,0.98),rgba(6,13,24,0.95));'
                    f'border:1px solid {color}1a;'
                    f'box-shadow:0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 {color}0e;'
                    f'animation:cardReveal 0.6s {delay} ease both;opacity:0;animation-fill-mode:both;">'
                    f'<div style="position:absolute;top:0;left:0;right:0;height:1px;'
                    f'background:linear-gradient(90deg,transparent,{color}50,transparent);"></div>'
                    f'<div style="position:absolute;top:8px;left:8px;width:14px;height:14px;'
                    f'border-top:1px solid {color}40;border-left:1px solid {color}40;"></div>'
                    f'<div style="position:absolute;bottom:8px;right:8px;width:14px;height:14px;'
                    f'border-bottom:1px solid {color}40;border-right:1px solid {color}40;"></div>'
                    f'<div style="font-size:28px;margin-bottom:14px;line-height:1;opacity:0.9;">{icon}</div>'
                    f'<div style="font-size:9.5px;font-weight:900;color:{color};letter-spacing:2.5px;'
                    f'font-family:Orbitron,monospace;line-height:1.4;margin-bottom:10px;">{title_}</div>'
                    f'<p style="font-size:8.5px;color:rgba(200,214,232,0.28);line-height:1.8;margin:0 0 16px;'
                    f'font-family:JetBrains Mono,monospace;">{desc}</p>'
                    f'<div style="font-size:7px;letter-spacing:2px;color:{color}50;font-family:Orbitron,monospace;">OPEN →</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                dest = 'combinatorics' if pid == 'comb' else pid
                if st.button('enter', key=f'_nb_{pid}', use_container_width=True): _go(dest)

    # ── CLT LIVE DEMO ──
    section_divider("#ff2d78", "#ffb700", "· · ·", anchor_id="sec-clt")
    st.markdown("""
    <div style="text-align:center;padding:8px 0 20px;position:relative;z-index:10;" class="plab-reveal" data-reveal-delay="0.1">
      <div style="font-size:8px;letter-spacing:7px;color:rgba(255,45,120,0.6);
           font-family:Orbitron,monospace;margin-bottom:10px;">INTERACTIVE DEMO</div>
      <div style="font-size:1.4rem;font-weight:900;
           background:linear-gradient(90deg,#ff2d78,#ffb700);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-family:Orbitron,monospace;letter-spacing:3px;">CENTRAL LIMIT THEOREM</div>
      <p style="font-size:10px;color:rgba(200,214,232,0.5);font-family:JetBrains Mono,monospace;
           margin-top:8px;letter-spacing:1px;">Watch sample means converge to Normal as n grows</p>
    </div>
    """, unsafe_allow_html=True)

    _, clt_mid, _ = st.columns([1, 10, 1])
    with clt_mid:
        clt_c1, clt_c2 = st.columns([1, 2.5], gap="large")
        with clt_c1:
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            clt_dist = st.selectbox("Source Distribution", ["Uniform", "Exponential", "Bimodal", "Skewed"], key="clt_dist")
            clt_n    = st.slider("Sample size n", 1, 200, 30, key="clt_n")
            clt_sims = st.slider("Simulations", 200, 5000, 2000, 200, key="clt_sims")
            clt_bins = st.slider("Histogram bins", 10, 60, 30, key="clt_bins")
            animate_clt = st.button("▶  ANIMATE CLT", key="clt_anim", use_container_width=True)
            st.markdown(
                f'<div style="margin-top:14px;padding:14px 16px;border-radius:10px;'
                f'background:rgba(255,45,120,0.05);border:1px solid rgba(255,45,120,0.2);">'
                f'<div style="font-size:7px;letter-spacing:4px;color:rgba(255,45,120,0.6);font-family:Orbitron,monospace;margin-bottom:6px;">CLT FORMULA</div>'
                f'<div style="font-size:11px;color:rgba(255,45,120,0.85);font-family:JetBrains Mono,monospace;line-height:2;">'
                f'X̄ ~ N(μ, σ²/n)<br>'
                f'<span style="color:rgba(200,214,232,0.4);font-size:9px;">as n → ∞</span></div></div>',
                unsafe_allow_html=True
            )

        with clt_c2:
            # Generate source samples
            np.random.seed(42)
            if clt_dist == "Uniform":
                raw = np.random.uniform(0, 1, (clt_sims, clt_n))
                mu_true, sig_true = 0.5, np.sqrt(1/12)
                src_label = "Uniform(0,1)"
            elif clt_dist == "Exponential":
                raw = np.random.exponential(1.0, (clt_sims, clt_n))
                mu_true, sig_true = 1.0, 1.0
                src_label = "Exponential(λ=1)"
            elif clt_dist == "Bimodal":
                raw = np.where(np.random.rand(clt_sims, clt_n) > 0.5,
                               np.random.normal(-2, 0.5, (clt_sims, clt_n)),
                               np.random.normal(2, 0.5, (clt_sims, clt_n)))
                mu_true, sig_true = 0.0, np.sqrt(4 + 0.25)
                src_label = "Bimodal"
            else:  # Skewed
                raw = np.random.gamma(2, 1, (clt_sims, clt_n))
                mu_true, sig_true = 2.0, np.sqrt(2.0)
                src_label = "Gamma(2,1) — Skewed"

            means = raw.mean(axis=1)
            clt_mean_se = sig_true / np.sqrt(clt_n)

            if animate_clt:
                ph = st.empty()
                steps = list(range(50, clt_sims + 1, max(50, clt_sims // 40)))
                if steps[-1] != clt_sims: steps.append(clt_sims)
                for k in steps:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                    # Left: source dist (single sample)
                    ax0 = axes[0]
                    src_single = raw[0]
                    ax0.hist(src_single, bins=20, color="#ff2d78", alpha=0.7, density=True, edgecolor="none")
                    ax0.set_title(f"Source: {src_label}", color="#ff2d78", fontsize=8, fontweight="bold")
                    ax0.set_ylabel("density", fontsize=7)
                    ax0.spines["top"].set_visible(False); ax0.spines["right"].set_visible(False)
                    # Right: sampling dist of means
                    ax1 = axes[1]
                    ax1.hist(means[:k], bins=clt_bins, color="#ffb700", alpha=0.75, density=True, edgecolor="none")
                    xr = np.linspace(means.min(), means.max(), 200)
                    ax1.plot(xr, stats.norm.pdf(xr, mu_true, clt_mean_se), color="#00f5ff", lw=2, label="N(μ, σ²/n)")
                    ax1.set_title(f"Sample Means (n={clt_n}, sims={k:,})", color="#ffb700", fontsize=8, fontweight="bold")
                    ax1.legend(fontsize=7, framealpha=0.1, labelcolor="#6a8aaa", facecolor="#000", edgecolor="#0a1628")
                    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
                    fig.tight_layout(pad=1.5)
                    ph.pyplot(fig, use_container_width=True); plt.close(fig)
                    time.sleep(0.04)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                ax0 = axes[0]
                ax0.hist(raw[0], bins=20, color="#ff2d78", alpha=0.7, density=True, edgecolor="none")
                xsrc = np.linspace(raw[0].min(), raw[0].max(), 200)
                ax0.set_title(f"Source: {src_label}", color="#ff2d78", fontsize=8, fontweight="bold")
                ax0.set_ylabel("density", fontsize=7)
                ax0.spines["top"].set_visible(False); ax0.spines["right"].set_visible(False)

                ax1 = axes[1]
                ax1.hist(means, bins=clt_bins, color="#ffb700", alpha=0.75, density=True, edgecolor="none")
                xr = np.linspace(means.min(), means.max(), 200)
                ax1.plot(xr, stats.norm.pdf(xr, mu_true, clt_mean_se), color="#00f5ff", lw=2.5, label=f"N({mu_true:.2f}, {clt_mean_se:.3f}²)")
                ax1.axvline(means.mean(), color="#00ff9d", lw=1.5, ls="--", alpha=0.8, label=f"x̄={means.mean():.3f}")
                ax1.set_title(f"Sampling Distribution of X̄  (n={clt_n}, {clt_sims:,} sims)", color="#ffb700", fontsize=8, fontweight="bold")
                ax1.legend(fontsize=7, framealpha=0.1, labelcolor="#6a8aaa", facecolor="#000", edgecolor="#0a1628")
                ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
                fig.tight_layout(pad=1.5)
                st.pyplot(fig, use_container_width=True); plt.close(fig)

            # stats row
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col_, lbl, val, clr in [
                (sc1, "SAMPLE MEAN", f"{means.mean():.4f}", "#00f5ff"),
                (sc2, "THEORETICAL μ", f"{mu_true:.4f}", "#ff2d78"),
                (sc3, "SAMPLE SE", f"{means.std():.4f}", "#ffb700"),
                (sc4, "THEORETICAL σ/√n", f"{clt_mean_se:.4f}", "#00ff9d"),
            ]:
                with col_:
                    st.markdown(
                        f'<div style="padding:10px 14px;border-radius:8px;text-align:center;'
                        f'background:rgba(6,13,24,0.8);border:1px solid {clr}22;margin-top:8px;">'
                        f'<div style="font-size:6.5px;letter-spacing:3px;color:{clr}66;font-family:Orbitron,monospace;margin-bottom:4px;">{lbl}</div>'
                        f'<div style="font-size:1.1rem;font-weight:900;color:{clr};font-family:Orbitron,monospace;'
                        f'text-shadow:0 0 12px {clr}55;">{val}</div></div>',
                        unsafe_allow_html=True
                    )

    # ── DISTRIBUTION COMPARATOR ──
    section_divider("#9d4edd", "#00f5ff", "· · ·", anchor_id="sec-compare")
    st.markdown("""
    <div style="text-align:center;padding:8px 0 20px;position:relative;z-index:10;" class="plab-reveal" data-reveal-delay="0.1">
      <div style="font-size:8px;letter-spacing:7px;color:rgba(157,78,221,0.6);
           font-family:Orbitron,monospace;margin-bottom:10px;">SIDE BY SIDE</div>
      <div style="font-size:1.4rem;font-weight:900;
           background:linear-gradient(90deg,#9d4edd,#00f5ff);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-family:Orbitron,monospace;letter-spacing:3px;">DISTRIBUTION COMPARATOR</div>
      <p style="font-size:10px;color:rgba(200,214,232,0.5);font-family:JetBrains Mono,monospace;
           margin-top:8px;letter-spacing:1px;">Overlay any two distributions and inspect their differences</p>
    </div>
    """, unsafe_allow_html=True)

    _, cmp_mid, _ = st.columns([1, 10, 1])
    with cmp_mid:
        COMP_DISTS = {
            "Normal(μ,σ)":      {"params": [("μ", -5.0, 5.0, 0.0, 0.1), ("σ", 0.1, 5.0, 1.0, 0.1)], "color": "#ffb700"},
            "Exponential(λ)":   {"params": [("λ", 0.1, 5.0, 1.0, 0.1)],                               "color": "#ff6b35"},
            "Gamma(k,θ)":       {"params": [("k", 0.5, 10.0, 2.0, 0.5), ("θ", 0.1, 5.0, 1.0, 0.1)],  "color": "#ff2d78"},
            "Beta(α,β)":        {"params": [("α", 0.5, 10.0, 2.0, 0.5), ("β", 0.5, 10.0, 3.0, 0.5)], "color": "#ff8c42"},
            "Chi-Squared(df)":  {"params": [("df", 1, 20, 5, 1)],                                     "color": "#c084fc"},
            "Student-t(df)":    {"params": [("df", 1, 30, 10, 1)],                                    "color": "#38bdf8"},
            "Uniform(a,b)":     {"params": [("a", -5.0, 0.0, 0.0, 0.5), ("b", 0.1, 10.0, 5.0, 0.5)], "color": "#00d4aa"},
            "Binomial(n,p)":    {"params": [("n", 5, 50, 20, 1), ("p", 0.01, 0.99, 0.5, 0.01)],      "color": "#00f5ff"},
            "Poisson(λ)":       {"params": [("λ", 0.5, 20.0, 5.0, 0.5)],                              "color": "#00ff9d"},
        }

        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            d1_name = st.selectbox("Distribution A", list(COMP_DISTS.keys()), index=0, key="cmp_d1")
            d1 = COMP_DISTS[d1_name]
            d1_params = {}
            for (pname, mn, mx, dv, stp) in d1["params"]:
                d1_params[pname] = st.slider(f"A · {pname}", float(mn) if isinstance(dv,float) else int(mn),
                                              float(mx) if isinstance(dv,float) else int(mx), dv,
                                              float(stp) if isinstance(dv,float) else int(stp), key=f"cmp_d1_{pname}")
        with cc2:
            d2_name = st.selectbox("Distribution B", list(COMP_DISTS.keys()), index=2, key="cmp_d2")
            d2 = COMP_DISTS[d2_name]
            d2_params = {}
            for (pname, mn, mx, dv, stp) in d2["params"]:
                d2_params[pname] = st.slider(f"B · {pname}", float(mn) if isinstance(dv,float) else int(mn),
                                              float(mx) if isinstance(dv,float) else int(mx), dv,
                                              float(stp) if isinstance(dv,float) else int(stp), key=f"cmp_d2_{pname}")

        # Build x range and PDFs
        def get_pdf(name, params):
            p = list(params.values())
            if name == "Normal(μ,σ)":
                x = np.linspace(p[0]-4*p[1], p[0]+4*p[1], 500)
                return x, stats.norm.pdf(x, p[0], p[1]), False, p[0], p[1]**2
            elif name == "Exponential(λ)":
                x = np.linspace(0, 6/p[0], 500)
                return x, stats.expon.pdf(x, scale=1/p[0]), False, 1/p[0], 1/p[0]**2
            elif name == "Gamma(k,θ)":
                x = np.linspace(0, p[0]*p[1]+5*np.sqrt(p[0])*p[1], 500)
                return x, stats.gamma.pdf(x, p[0], scale=p[1]), False, p[0]*p[1], p[0]*p[1]**2
            elif name == "Beta(α,β)":
                x = np.linspace(0.001, 0.999, 500)
                return x, stats.beta.pdf(x, p[0], p[1]), False, p[0]/(p[0]+p[1]), p[0]*p[1]/((p[0]+p[1])**2*(p[0]+p[1]+1))
            elif name == "Chi-Squared(df)":
                x = np.linspace(0.01, p[0]+5*np.sqrt(2*p[0]), 500)
                return x, stats.chi2.pdf(x, int(p[0])), False, p[0], 2*p[0]
            elif name == "Student-t(df)":
                x = np.linspace(-5, 5, 500)
                return x, stats.t.pdf(x, int(p[0])), False, 0, int(p[0])/(int(p[0])-2) if p[0]>2 else float('inf')
            elif name == "Uniform(a,b)":
                x = np.linspace(p[0]-0.5, p[1]+0.5, 500)
                return x, stats.uniform.pdf(x, p[0], p[1]-p[0]), False, (p[0]+p[1])/2, (p[1]-p[0])**2/12
            elif name == "Binomial(n,p)":
                x = np.arange(0, int(p[0])+1)
                return x, stats.binom.pmf(x, int(p[0]), p[1]), True, int(p[0])*p[1], int(p[0])*p[1]*(1-p[1])
            elif name == "Poisson(λ)":
                x = np.arange(0, int(p[0]*3)+1)
                return x, stats.poisson.pmf(x, p[0]), True, p[0], p[0]
            return np.array([0,1]), np.array([0,1]), False, 0, 0

        x1, y1, disc1, mu1, var1 = get_pdf(d1_name, d1_params)
        x2, y2, disc2, mu2, var2 = get_pdf(d2_name, d2_params)

        fig, ax = plt.subplots(figsize=(12, 4))
        if disc1:
            ax.bar(x1, y1, alpha=0.55, color=d1["color"], label=d1_name, width=0.4, align='center')
        else:
            ax.fill_between(x1, y1, alpha=0.18, color=d1["color"])
            ax.plot(x1, y1, color=d1["color"], lw=2.5, label=d1_name)
        if disc2:
            ax.bar(x2 + (0.4 if disc1 else 0), y2, alpha=0.55, color=d2["color"], label=d2_name, width=0.4, align='center')
        else:
            ax.fill_between(x2, y2, alpha=0.18, color=d2["color"])
            ax.plot(x2, y2, color=d2["color"], lw=2.5, label=d2_name, linestyle="--")

        ax.axvline(mu1, color=d1["color"], lw=1.2, ls=":", alpha=0.7, label=f"μ_A={mu1:.3f}")
        ax.axvline(mu2, color=d2["color"], lw=1.2, ls=":", alpha=0.7, label=f"μ_B={mu2:.3f}")
        ax.set_title("Distribution Comparator", color="#9d4edd", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.12, labelcolor="#c8d6e8", facecolor="#000", edgecolor="#0a1628", ncol=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # stats comparison row
        sm1, sm2, sm3, sm4 = st.columns(4)
        for col_, lbl, val_a, val_b, ca, cb in [
            (sm1, "MEAN",     f"{mu1:.4f}", f"{mu2:.4f}", d1["color"], d2["color"]),
            (sm2, "VARIANCE", f"{var1:.4f}", f"{var2:.4f}", d1["color"], d2["color"]),
            (sm3, "STD DEV",  f"{np.sqrt(var1):.4f}", f"{np.sqrt(var2):.4f}", d1["color"], d2["color"]),
            (sm4, "TYPE",     "Discrete" if disc1 else "Continuous", "Discrete" if disc2 else "Continuous", d1["color"], d2["color"]),
        ]:
            with col_:
                st.markdown(
                    f'<div style="padding:10px 14px;border-radius:8px;margin-top:8px;'
                    f'background:rgba(6,13,24,0.8);border:1px solid rgba(255,255,255,0.08);">'
                    f'<div style="font-size:6.5px;letter-spacing:3px;color:rgba(200,214,232,0.4);font-family:Orbitron,monospace;margin-bottom:6px;">{lbl}</div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="font-size:10px;font-weight:700;color:{ca};font-family:JetBrains Mono,monospace;">A: {val_a}</span>'
                    f'<span style="font-size:10px;font-weight:700;color:{cb};font-family:JetBrains Mono,monospace;">B: {val_b}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

    # ─ Project Credit — rendered via _stc.html() so no escaping issues
    section_divider("#00ff9d", "#9d4edd", "· · ·", anchor_id="sec-quiz")
    st.markdown("""
    <div style="text-align:center;padding:8px 0 20px;position:relative;z-index:10;" class="plab-reveal" data-reveal-delay="0.1">
      <div style="font-size:8px;letter-spacing:7px;color:rgba(0,255,157,0.6);
           font-family:Orbitron,monospace;margin-bottom:10px;">TEST YOUR KNOWLEDGE</div>
      <div style="font-size:1.4rem;font-weight:900;
           background:linear-gradient(90deg,#00ff9d,#00f5ff);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-family:Orbitron,monospace;letter-spacing:3px;">PROBABILITY QUIZ</div>
      <p style="font-size:10px;color:rgba(200,214,232,0.5);font-family:JetBrains Mono,monospace;
           margin-top:8px;letter-spacing:1px;">12 questions · Score yourself · Instant feedback</p>
    </div>
    """, unsafe_allow_html=True)
    _stc.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:transparent;font-family:'Space Grotesk',sans-serif;}
.quiz-wrap{max-width:860px;margin:0 auto;padding:0 24px 32px;}
.quiz-box{padding:28px 32px;border-radius:18px;
  background:linear-gradient(135deg,rgba(6,13,24,0.97),rgba(10,22,40,0.95));
  border:1px solid rgba(0,245,255,0.18);position:relative;overflow:hidden;}
.quiz-box::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(0,245,255,0.5),rgba(157,78,221,0.5),transparent);}
.corner-tl{position:absolute;top:10px;left:10px;width:16px;height:16px;border-top:1px solid rgba(0,245,255,0.4);border-left:1px solid rgba(0,245,255,0.4);}
.corner-br{position:absolute;bottom:10px;right:10px;width:16px;height:16px;border-bottom:1px solid rgba(157,78,221,0.4);border-right:1px solid rgba(157,78,221,0.4);}
.quiz-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
.quiz-label{font-size:7px;letter-spacing:5px;color:rgba(0,245,255,0.5);font-family:Orbitron,monospace;}
.quiz-progress{font-size:7px;letter-spacing:3px;color:rgba(157,78,221,0.6);font-family:'JetBrains Mono',monospace;}
.quiz-score-display{font-size:7px;letter-spacing:3px;color:rgba(0,255,157,0.6);font-family:'JetBrains Mono',monospace;}
.progress-bar-track{height:2px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;margin-bottom:14px;}
.progress-bar-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#00f5ff,#9d4edd);transition:width 0.4s ease;}
.quiz-question{font-size:13px;color:rgba(200,214,232,0.85);font-family:'JetBrains Mono',monospace;line-height:1.7;margin-bottom:18px;}
.quiz-options{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;}
.qbtn{padding:10px 16px;border-radius:8px;font-size:9px;font-family:Orbitron,monospace;letter-spacing:1.5px;cursor:pointer;transition:all 0.2s;text-align:left;line-height:1.5;}
.qbtn-default{background:rgba(0,245,255,0.04);border:1px solid rgba(0,245,255,0.2);color:rgba(200,214,232,0.75);}
.qbtn-default:hover{background:rgba(0,245,255,0.12);border-color:rgba(0,245,255,0.5);color:#fff;box-shadow:0 0 12px rgba(0,245,255,0.15);}
.qbtn-correct{background:rgba(0,255,157,0.12)!important;border:1px solid rgba(0,255,157,0.6)!important;color:#00ff9d!important;box-shadow:0 0 14px rgba(0,255,157,0.2);}
.qbtn-wrong{background:rgba(255,45,120,0.10)!important;border:1px solid rgba(255,45,120,0.5)!important;color:#ff2d78!important;}
.qbtn-reveal{background:rgba(0,255,157,0.07)!important;border:1px solid rgba(0,255,157,0.35)!important;color:rgba(0,255,157,0.7)!important;}
.qbtn:disabled{cursor:default;}
#quiz-result{font-size:10px;font-family:'JetBrains Mono',monospace;min-height:22px;margin-bottom:8px;transition:all 0.3s;}
.quiz-nav{display:flex;justify-content:space-between;align-items:center;margin-top:4px;}
.nav-btn{padding:7px 20px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;letter-spacing:2px;cursor:pointer;border:1px solid rgba(157,78,221,0.35);background:rgba(157,78,221,0.07);color:rgba(157,78,221,0.85);transition:all 0.2s;}
.nav-btn:hover{background:rgba(157,78,221,0.18);border-color:rgba(157,78,221,0.65);box-shadow:0 0 12px rgba(157,78,221,0.2);}
.nav-btn:disabled{opacity:0.3;cursor:default;}
.summary-box{padding:20px 24px;border-radius:14px;background:rgba(0,255,157,0.05);border:1px solid rgba(0,255,157,0.2);text-align:center;}
.summary-score{font-size:2.4rem;font-weight:900;font-family:Orbitron,monospace;color:#00ff9d;text-shadow:0 0 20px rgba(0,255,157,0.5);}
.summary-label{font-size:8px;letter-spacing:4px;color:rgba(0,255,157,0.5);font-family:Orbitron,monospace;margin-top:4px;margin-bottom:12px;}
.restart-btn{padding:9px 28px;border-radius:8px;font-size:8px;font-family:Orbitron,monospace;letter-spacing:3px;cursor:pointer;border:1px solid rgba(0,245,255,0.4);background:rgba(0,245,255,0.07);color:rgba(0,245,255,0.9);transition:all 0.2s;}
.restart-btn:hover{background:rgba(0,245,255,0.18);box-shadow:0 0 14px rgba(0,245,255,0.25);}
.tag{display:inline-block;padding:2px 10px;border-radius:4px;font-size:7px;letter-spacing:2px;font-family:'JetBrains Mono',monospace;margin-bottom:8px;background:rgba(255,183,0,0.08);border:1px solid rgba(255,183,0,0.25);color:rgba(255,183,0,0.7);}
</style>
<div class="quiz-wrap">
  <div class="quiz-box">
    <div class="corner-tl"></div>
    <div class="corner-br"></div>
    <div class="quiz-header">
      <div class="quiz-label">⚡ QUICK CHALLENGE</div>
      <div style="display:flex;gap:16px;align-items:center;">
        <div class="quiz-score-display" id="score-display">SCORE: 0/0</div>
        <div class="quiz-progress" id="q-counter">Q 1 / 12</div>
      </div>
    </div>
    <div class="progress-bar-track"><div class="progress-bar-fill" id="prog-bar" style="width:8.33%"></div></div>
    <div id="quiz-main"></div>
  </div>
</div>
<script>
var QUESTIONS=[
  {tag:"INDEPENDENCE",q:"If P(A)=0.3 and P(B)=0.5, and A,B are independent — what is P(A∩B)?",opts:["0.15","0.80","0.20","0.50"],ans:0,exp:"P(A)·P(B) = 0.3 × 0.5 = 0.15"},
  {tag:"BAYES",q:"P(Disease)=0.01, P(+|Disease)=0.95, P(+|No Disease)=0.05. What is P(Disease|+)?",opts:["0.161","0.950","0.010","0.500"],ans:0,exp:"Bayes: (0.95×0.01)/((0.95×0.01)+(0.05×0.99)) ≈ 0.161"},
  {tag:"BINOMIAL",q:"X~Binomial(n=5, p=0.4). What is E[X]?",opts:["2.0","1.2","2.5","0.4"],ans:0,exp:"E[X] = np = 5 × 0.4 = 2.0"},
  {tag:"POISSON",q:"If X~Poisson(λ=3), what is Var(X)?",opts:["3","9","1.73","6"],ans:0,exp:"For Poisson, Var(X) = λ = 3"},
  {tag:"NORMAL",q:"For X~N(μ,σ²), roughly what % of data lies within ±2σ?",opts:["95.4%","68.3%","99.7%","50.0%"],ans:0,exp:"The empirical 68-95-99.7 rule: ±2σ ≈ 95.4%"},
  {tag:"COMBINATORICS",q:"How many ways can you choose 3 items from 8 (order doesn't matter)?",opts:["56","336","24","512"],ans:0,exp:"C(8,3) = 8!/(3!·5!) = 56"},
  {tag:"CONDITIONAL",q:"P(A)=0.6, P(B)=0.4, P(A∩B)=0.2. What is P(A|B)?",opts:["0.50","0.33","0.20","0.75"],ans:0,exp:"P(A|B) = P(A∩B)/P(B) = 0.2/0.4 = 0.50"},
  {tag:"GEOMETRIC",q:"X~Geometric(p=0.25). What is E[X] (number of trials until first success)?",opts:["4","0.25","3","2"],ans:0,exp:"E[X] = 1/p = 1/0.25 = 4"},
  {tag:"CLT",q:"The Central Limit Theorem states that the sample mean converges to which distribution as n→∞?",opts:["Normal","Poisson","Uniform","Exponential"],ans:0,exp:"CLT: sample mean → N(μ, σ²/n) regardless of the original distribution"},
  {tag:"CHI-SQUARE",q:"A Chi-Square test is primarily used for:",opts:["Goodness-of-fit & independence","Comparing two means","Estimating λ","Finding correlation"],ans:0,exp:"Chi-Square tests categorical data for goodness-of-fit or independence"},
  {tag:"BIRTHDAY",q:"In the Birthday Paradox, with how many people is P(shared birthday) > 50%?",opts:["23","50","100","10"],ans:0,exp:"With just 23 people, P(shared birthday) ≈ 50.7% — the famous Birthday Paradox!"},
  {tag:"EXPONENTIAL",q:"X~Exponential(λ=2). What is P(X > 1)?",opts:["e⁻² ≈ 0.135","e⁻¹ ≈ 0.368","0.5","1-e⁻²"],ans:0,exp:"P(X>t) = e^(-λt) = e^(-2×1) = e⁻² ≈ 0.135"},
];

var cur=0, score=0, answered=new Array(QUESTIONS.length).fill(null);

function render(){
  var qm=document.getElementById('quiz-main');
  if(cur>=QUESTIONS.length){
    // Summary
    var pct=Math.round(score/QUESTIONS.length*100);
    var grade=pct>=90?'EXCELLENT':pct>=70?'GOOD JOB':pct>=50?'KEEP PRACTICING':'KEEP GOING';
    var gc=pct>=90?'#00ff9d':pct>=70?'#00f5ff':pct>=50?'#ffb700':'#ff2d78';
    qm.innerHTML='<div class="summary-box"><div class="summary-score" style="color:'+gc+';text-shadow:0 0 20px '+gc+'80;">'+score+'/'+QUESTIONS.length+'</div>'
      +'<div class="summary-label">'+grade+'</div>'
      +'<div style="font-size:9px;color:rgba(200,214,232,0.55);font-family:JetBrains Mono,monospace;margin-bottom:16px;">'+pct+'% accuracy</div>'
      +'<button class="restart-btn" onclick="restart()">↺  RESTART QUIZ</button></div>';
    return;
  }
  var q=QUESTIONS[cur];
  var done=answered[cur]!==null;
  var html='<div class="tag">'+q.tag+'</div>';
  html+='<div class="quiz-question">'+q.q+'</div>';
  html+='<div class="quiz-options">';
  for(var i=0;i<q.opts.length;i++){
    var cls='qbtn ';
    if(done){
      if(i===q.ans) cls+='qbtn-correct';
      else if(i===answered[cur]) cls+='qbtn-wrong';
      else cls+='qbtn-default';
    } else { cls+='qbtn-default'; }
    html+='<button class="'+cls+'" '+(done?'disabled':'')+' onclick="pick('+i+')">'+q.opts[i]+'</button>';
  }
  html+='</div>';
  html+='<div id="quiz-result">';
  if(done){
    if(answered[cur]===q.ans){
      html+='<span style="color:#00ff9d;font-size:11px;">✓ Correct!&nbsp;&nbsp;'+q.exp+'</span>';
    } else {
      html+='<span style="color:#ff2d78;font-size:11px;">✗ '+q.exp+'</span>';
    }
  }
  html+='</div>';
  html+='<div class="quiz-nav">';
  html+='<button class="nav-btn" onclick="nav(-1)" '+(cur===0?'disabled':'')+'>← PREV</button>';
  html+='<span style="font-size:8px;color:rgba(200,214,232,0.3);font-family:JetBrains Mono,monospace;">'+(done?'answered':'unanswered')+'</span>';
  if(cur<QUESTIONS.length-1){
    html+='<button class="nav-btn" onclick="nav(1)">NEXT →</button>';
  } else {
    html+='<button class="nav-btn" onclick="finish()">FINISH ✓</button>';
  }
  html+='</div>';
  qm.innerHTML=html;
  // update progress
  document.getElementById('q-counter').textContent='Q '+(cur+1)+' / '+QUESTIONS.length;
  document.getElementById('prog-bar').style.width=(((cur+1)/QUESTIONS.length)*100)+'%';
  document.getElementById('score-display').textContent='SCORE: '+score+'/'+QUESTIONS.length;
}
function pick(i){
  if(answered[cur]!==null)return;
  var q=QUESTIONS[cur];
  answered[cur]=i;
  if(i===q.ans) score++;
  render();
}
function nav(d){cur=Math.max(0,Math.min(QUESTIONS.length-1,cur+d));render();}
function finish(){cur=QUESTIONS.length;render();}
function restart(){cur=0;score=0;answered=new Array(QUESTIONS.length).fill(null);render();}
render();
</script>
""", height=300, scrolling=False)

    section_divider("#00f5ff", "#9d4edd", "· · ·")
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    _stc.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{background:transparent;overflow:hidden;}

@keyframes aurora{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes shimmer{0%{background-position:-300% center}100%{background-position:300% center}}
@keyframes floatY{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes floatY2{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes progressBar{from{width:0}to{width:var(--w,100%)}}
@keyframes pulseGlow{0%,100%{opacity:0.6}50%{opacity:1}}
@keyframes scanLine{0%{top:-2px;opacity:0}10%{opacity:1}90%{opacity:0.6}100%{top:100%;opacity:0}}
@keyframes borderFlow{0%,100%{opacity:0.4}50%{opacity:0.9}}
@keyframes countUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes glitch{0%,100%{clip-path:none}92%{clip-path:none}93%{clip-path:inset(20% 0 60% 0);transform:translateX(-3px)}94%{clip-path:inset(50% 0 30% 0);transform:translateX(3px)}95%{clip-path:none;transform:translateX(0)}}
@keyframes orbPulse{0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(0,245,255,0.4)}50%{transform:scale(1.08);box-shadow:0 0 0 8px rgba(0,245,255,0)}}
@keyframes rotate{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

.fw{
  font-family:'Space Grotesk',sans-serif;
  position:relative;overflow:hidden;
  background:linear-gradient(180deg,rgba(2,5,12,0.99) 0%,rgba(0,0,0,1) 100%);
  padding:0;
}

/* ── Top aurora bar ── */
.aurora-top{
  position:relative;height:3px;
  background:linear-gradient(90deg,#00f5ff,#9d4edd,#ff2d78,#ffb700,#00ff9d,#00f5ff);
  background-size:300%;animation:aurora 3s linear infinite;
  box-shadow:0 0 24px rgba(0,245,255,0.4),0 0 60px rgba(157,78,221,0.2);
}

/* ── Ambient glows ── */
.g1{position:absolute;top:-120px;left:-60px;width:500px;height:500px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,245,255,0.055),transparent 65%);filter:blur(50px);pointer-events:none;}
.g2{position:absolute;bottom:-100px;right:-40px;width:450px;height:450px;border-radius:50%;
  background:radial-gradient(circle,rgba(157,78,221,0.055),transparent 65%);filter:blur(50px);pointer-events:none;}
.g3{position:absolute;top:40%;left:50%;transform:translate(-50%,-50%);width:600px;height:200px;
  background:radial-gradient(ellipse,rgba(255,45,120,0.025),transparent 70%);filter:blur(40px);pointer-events:none;}

/* ── Scan line effect ── */
.scan{position:absolute;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,rgba(0,245,255,0.15),transparent);
  animation:scanLine 5s ease-in-out infinite;pointer-events:none;z-index:0;}

/* ── Grid overlay ── */
.grid-bg{position:absolute;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(0,245,255,0.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,245,255,0.025) 1px,transparent 1px);
  background-size:60px 60px;}

/* ── Hero quote band ── */
.quote-band{
  padding:52px 8vw 44px;
  text-align:center;position:relative;z-index:2;
  border-bottom:1px solid rgba(255,255,255,0.05);
}
.quote-text{
  font-family:Orbitron,monospace;font-size:clamp(1.1rem,2.5vw,1.8rem);
  font-weight:900;letter-spacing:3px;line-height:1.3;
  background:linear-gradient(135deg,#a8f4ff 0%,#00f5ff 25%,#fff 50%,#bf5fff 70%,#ff2d78 100%);
  background-size:300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shimmer 6s ease infinite;
}
.quote-sub{
  margin-top:12px;font-size:11px;letter-spacing:5px;
  color:rgba(200,214,232,0.35);font-family:Orbitron,monospace;
}

/* ── Stats band ── */
.stats-band{
  display:flex;justify-content:center;align-items:stretch;
  gap:0;border-bottom:1px solid rgba(255,255,255,0.05);
  position:relative;z-index:2;
}
.stat-item{
  flex:1;padding:28px 20px;text-align:center;
  border-right:1px solid rgba(255,255,255,0.06);
  position:relative;overflow:hidden;
  transition:background 0.3s;
}
.stat-item:last-child{border-right:none;}
.stat-item:hover{background:rgba(255,255,255,0.02);}
.stat-num{
  font-family:Orbitron,monospace;font-size:1.9rem;font-weight:900;
  line-height:1;margin-bottom:6px;
}
.stat-lbl{font-size:7px;letter-spacing:4px;color:rgba(200,214,232,0.35);font-family:Orbitron,monospace;}
.stat-bar{height:2px;border-radius:1px;margin-top:10px;animation:progressBar 1.8s ease both;}

/* ── Main body ── */
.body-grid{
  display:grid;grid-template-columns:1.3fr 1px 1fr 1px 1fr;
  gap:0;padding:48px 8vw 52px;
  position:relative;z-index:2;
  border-bottom:1px solid rgba(255,255,255,0.05);
}
.col-div{background:linear-gradient(180deg,transparent,rgba(255,255,255,0.1),transparent);}

/* Shared col styles */
.col{padding:0 36px;}
.col:first-child{padding-left:0;}
.col:last-child{padding-right:0;}
.col-lbl{font-size:7px;letter-spacing:6px;color:rgba(255,255,255,0.4);
  font-family:Orbitron,monospace;margin-bottom:18px;display:flex;align-items:center;gap:8px;}
.col-lbl::before{content:'';width:16px;height:1px;background:currentColor;opacity:0.5;}

/* Identity col */
.name{
  font-family:Orbitron,monospace;font-size:2rem;font-weight:900;
  letter-spacing:2px;line-height:1;margin-bottom:8px;
  background:linear-gradient(90deg,#00f5ff 0%,#9d4edd 45%,#ff2d78 90%);
  background-size:200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shimmer 5s ease infinite,glitch 8s ease-in-out infinite;
}
.sub-name{font-size:8.5px;color:rgba(200,214,232,0.55);letter-spacing:3px;margin-bottom:20px;font-family:Orbitron,monospace;}
.badges{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:28px;}
.badge{padding:4px 13px;border-radius:20px;font-size:7px;letter-spacing:2px;font-family:Orbitron,monospace;
  transition:all 0.25s;}
.b1{color:rgba(0,245,255,0.9);border:1px solid rgba(0,245,255,0.4);background:rgba(0,245,255,0.08);}
.b1:hover{background:rgba(0,245,255,0.15);box-shadow:0 0 12px rgba(0,245,255,0.3);}
.b2{color:rgba(157,78,221,0.9);border:1px solid rgba(157,78,221,0.4);background:rgba(157,78,221,0.08);}
.b2:hover{background:rgba(157,78,221,0.15);box-shadow:0 0 12px rgba(157,78,221,0.3);}
.b3{color:rgba(255,183,0,0.9);border:1px solid rgba(255,183,0,0.4);background:rgba(255,183,0,0.08);}
.b3:hover{background:rgba(255,183,0,0.15);box-shadow:0 0 12px rgba(255,183,0,0.3);}

/* Status bars */
.status-lbl{font-size:7px;letter-spacing:4px;color:rgba(255,255,255,0.35);
  font-family:Orbitron,monospace;margin-bottom:12px;}
.bar-item{margin-bottom:10px;}
.bar-row{display:flex;justify-content:space-between;margin-bottom:4px;}
.bar-row span{font-size:7.5px;font-family:'JetBrains Mono',monospace;}
.bar-track{height:3px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;position:relative;}
.bar-fill{height:100%;border-radius:2px;animation:progressBar 1.6s cubic-bezier(0.23,1,0.32,1) both;}
.bar-glow{position:absolute;top:-2px;right:0;width:6px;height:7px;border-radius:50%;filter:blur(3px);}

/* Facts col */
.fact-card{
  padding:18px 20px;border-radius:14px;margin-bottom:14px;
  border:1px solid rgba(255,255,255,0.08);
  background:linear-gradient(135deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01));
  position:relative;overflow:hidden;transition:border-color 0.3s,box-shadow 0.3s;
}
.fact-card:hover{border-color:rgba(0,245,255,0.2);box-shadow:0 0 20px rgba(0,245,255,0.07);}
.fact-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(0,245,255,0.2),transparent);}
.fact-icon{font-size:16px;margin-bottom:8px;}
.fact-text{font-size:9.5px;color:rgba(200,214,232,0.75);line-height:1.75;}
.fact-hl{font-weight:700;}
.tech-grid{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px;}
.tech{padding:4px 11px;border-radius:6px;font-size:7px;font-family:'JetBrains Mono',monospace;
  transition:all 0.2s;}
.t1{color:rgba(255,183,0,0.9);border:1px solid rgba(255,183,0,0.3);background:rgba(255,183,0,0.07);}
.t2{color:rgba(255,45,120,0.9);border:1px solid rgba(255,45,120,0.3);background:rgba(255,45,120,0.07);}
.t3{color:rgba(0,245,255,0.9);border:1px solid rgba(0,245,255,0.3);background:rgba(0,245,255,0.07);}
.t4{color:rgba(157,78,221,0.9);border:1px solid rgba(157,78,221,0.3);background:rgba(157,78,221,0.07);}
.t5{color:rgba(0,255,157,0.9);border:1px solid rgba(0,255,157,0.3);background:rgba(0,255,157,0.07);}
.tech:hover{transform:translateY(-2px);filter:brightness(1.3);}

/* Connect col */
.social-card{
  display:flex;align-items:center;gap:14px;
  padding:14px 18px;border-radius:14px;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.09);
  text-decoration:none;
  transition:all 0.28s ease;
  cursor:pointer;margin-bottom:12px;
  position:relative;overflow:hidden;
}
.social-card::after{content:'';position:absolute;inset:0;opacity:0;
  background:linear-gradient(135deg,rgba(255,255,255,0.04),transparent);
  transition:opacity 0.3s;}
.social-card:hover::after{opacity:1;}
.social-card.gh:hover{border-color:rgba(0,245,255,0.4);box-shadow:0 8px 28px rgba(0,245,255,0.12),inset 0 1px 0 rgba(0,245,255,0.1);}
.social-card.li:hover{border-color:rgba(157,78,221,0.4);box-shadow:0 8px 28px rgba(157,78,221,0.12),inset 0 1px 0 rgba(157,78,221,0.1);}
.social-icon{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;border:1px solid;position:relative;}
.gh .social-icon{background:rgba(0,245,255,0.07);border-color:rgba(0,245,255,0.2);}
.li .social-icon{background:rgba(157,78,221,0.07);border-color:rgba(157,78,221,0.2);}
.s-meta{flex:1;}
.s-label{font-size:7px;letter-spacing:3px;color:rgba(200,214,232,0.45);
  font-family:Orbitron,monospace;margin-bottom:3px;}
.s-url{font-size:10px;font-weight:600;font-family:'JetBrains Mono',monospace;}
.gh-url{color:rgba(0,245,255,0.9);}
.li-url{color:rgba(157,78,221,0.9);}
.s-arrow{color:rgba(255,255,255,0.2);font-size:16px;transition:transform 0.2s,color 0.2s;}
.social-card:hover .s-arrow{transform:translateX(3px);color:rgba(255,255,255,0.5);}

/* Availability badge */
.avail{display:flex;align-items:center;gap:10px;padding:12px 16px;
  border-radius:10px;background:rgba(0,255,157,0.05);
  border:1px solid rgba(0,255,157,0.2);margin-top:4px;}
.avail-dot{width:7px;height:7px;border-radius:50%;background:#00ff9d;
  box-shadow:0 0 8px #00ff9d;animation:orbPulse 2s infinite;flex-shrink:0;}
.avail-text{font-size:8px;color:rgba(0,255,157,0.8);letter-spacing:2px;font-family:Orbitron,monospace;}

/* Bottom copyright bar */
.copy-bar{
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
  padding:18px 8vw;position:relative;z-index:2;
}
.copy-left{font-size:8px;color:rgba(200,214,232,0.3);letter-spacing:1.5px;font-family:'JetBrains Mono',monospace;}
.copy-right{display:flex;align-items:center;gap:6px;}
.copy-dot{width:4px;height:4px;border-radius:50%;}
.copy-version{font-size:7.5px;color:rgba(200,214,232,0.25);letter-spacing:2px;font-family:Orbitron,monospace;}
</style>

<div class="fw">
  <div class="aurora-top"></div>
  <div class="g1"></div><div class="g2"></div><div class="g3"></div>
  <div class="grid-bg"></div>
  <div class="scan"></div>

  <!-- ── QUOTE BAND ── -->
  <div class="quote-band">
    <div class="quote-text">"Probability is the language<br>the universe speaks in."</div>
    <div class="quote-sub">MASTER THE MATHEMATICS · UNDERSTAND THE UNCERTAINTY</div>
  </div>

  <!-- ── STATS BAND ── -->
  <div class="stats-band">
    <div class="stat-item">
      <div class="stat-num" style="color:#00f5ff;text-shadow:0 0 20px rgba(0,245,255,0.5);">10</div>
      <div class="stat-lbl">DISTRIBUTIONS</div>
      <div class="stat-bar" style="background:linear-gradient(90deg,#00f5ff,#9d4edd);--w:100%;"></div>
    </div>
    <div class="stat-item">
      <div class="stat-num" style="color:#9d4edd;text-shadow:0 0 20px rgba(157,78,221,0.5);">4</div>
      <div class="stat-lbl">STAT TOOLS</div>
      <div class="stat-bar" style="background:linear-gradient(90deg,#9d4edd,#ff2d78);--w:100%;animation-delay:0.15s;"></div>
    </div>
    <div class="stat-item">
      <div class="stat-num" style="color:#ff2d78;text-shadow:0 0 20px rgba(255,45,120,0.5);">12</div>
      <div class="stat-lbl">QUIZ QUESTIONS</div>
      <div class="stat-bar" style="background:linear-gradient(90deg,#ff2d78,#ffb700);--w:100%;animation-delay:0.3s;"></div>
    </div>
    <div class="stat-item">
      <div class="stat-num" style="color:#ffb700;text-shadow:0 0 20px rgba(255,183,0,0.5);">∞</div>
      <div class="stat-lbl">SIMULATIONS</div>
      <div class="stat-bar" style="background:linear-gradient(90deg,#ffb700,#00ff9d);--w:100%;animation-delay:0.45s;"></div>
    </div>
    <div class="stat-item">
      <div class="stat-num" style="color:#00ff9d;text-shadow:0 0 20px rgba(0,255,157,0.5);">98%</div>
      <div class="stat-lbl">ENGINE UPTIME</div>
      <div class="stat-bar" style="background:linear-gradient(90deg,#00ff9d,#00f5ff);--w:98%;animation-delay:0.6s;"></div>
    </div>
  </div>

  <!-- ── MAIN BODY ── -->
  <div class="body-grid">

    <!-- LEFT: Identity -->
    <div class="col" style="padding-left:0;">
      <div class="col-lbl">PROJECT BY</div>
      <div class="name">DESHAN GAUTAM</div>
      <div class="sub-name">FIRST YEAR &middot; IIT MADRAS &middot; BS DATA SCIENCE</div>
      <div class="badges">
        <span class="badge b1">FIRST YEAR</span>
        <span class="badge b2">IIT MADRAS</span>
        <span class="badge b3">BS DATA SCIENCE</span>
      </div>

      <div class="status-lbl">SYSTEM STATUS</div>
      <div class="bar-item">
        <div class="bar-row">
          <span style="color:rgba(0,245,255,0.7);">MONTE CARLO ENGINE</span>
          <span style="color:rgba(0,245,255,0.7);">98%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="--w:98%;background:linear-gradient(90deg,#00f5ff,#9d4edd);"></div>
        </div>
      </div>
      <div class="bar-item">
        <div class="bar-row">
          <span style="color:rgba(157,78,221,0.7);">DISTRIBUTION LIBRARY</span>
          <span style="color:rgba(157,78,221,0.7);">100%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="--w:100%;background:linear-gradient(90deg,#9d4edd,#ff2d78);animation-delay:0.2s;"></div>
        </div>
      </div>
      <div class="bar-item">
        <div class="bar-row">
          <span style="color:rgba(0,255,157,0.7);">QUIZ ENGINE</span>
          <span style="color:rgba(0,255,157,0.7);">100%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="--w:100%;background:linear-gradient(90deg,#00ff9d,#00f5ff);animation-delay:0.35s;"></div>
        </div>
      </div>
      <div class="bar-item">
        <div class="bar-row">
          <span style="color:rgba(255,183,0,0.7);">VISUALISATION LAYER</span>
          <span style="color:rgba(255,183,0,0.7);">95%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="--w:95%;background:linear-gradient(90deg,#ffb700,#ff2d78);animation-delay:0.5s;"></div>
        </div>
      </div>
    </div>

    <div class="col-div"></div>

    <!-- CENTER: Facts + Stack -->
    <div class="col">
      <div class="col-lbl">DID YOU KNOW</div>

      <div class="fact-card">
        <div class="fact-icon">🎂</div>
        <div class="fact-text">With just <span class="fact-hl" style="color:#00f5ff;">23 people</span> in a room, there is a <span class="fact-hl" style="color:#ff2d78;">50%+</span> chance two share a birthday — the <em>Birthday Paradox!</em></div>
      </div>

      <div class="fact-card" style="border-color:rgba(157,78,221,0.15);">
        <div class="fact-icon">🎲</div>
        <div class="fact-text">The <span class="fact-hl" style="color:#9d4edd;">Central Limit Theorem</span> says any distribution's sample means converge to a <em>Normal distribution</em> — nature's favourite bell curve.</div>
      </div>

      <div class="fact-card" style="border-color:rgba(255,183,0,0.15);">
        <div class="fact-icon">⚡</div>
        <div class="fact-text"><span class="fact-hl" style="color:#ffb700;">Bayes' Theorem</span> powers everything from spam filters to medical diagnosis — probability updated with evidence.</div>
      </div>

      <div class="col-lbl" style="margin-top:22px;">BUILT WITH</div>
      <div class="tech-grid">
        <span class="tech t1">Python</span>
        <span class="tech t2">Streamlit</span>
        <span class="tech t3">NumPy</span>
        <span class="tech t4">SciPy</span>
        <span class="tech t5">Matplotlib</span>
        <span class="tech t3" style="color:rgba(255,183,0,0.9);border-color:rgba(255,183,0,0.3);background:rgba(255,183,0,0.07);">Plotly</span>
      </div>
    </div>

    <div class="col-div"></div>

    <!-- RIGHT: Connect -->
    <div class="col" style="padding-right:0;">
      <div class="col-lbl">CONNECT</div>

      <a href="https://github.com/deshan-5" target="_blank" class="social-card gh float1">
        <div class="social-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="rgba(0,245,255,0.85)">
            <path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.868-.013-1.703-2.782.604-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836a9.59 9.59 0 012.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.202 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/>
          </svg>
        </div>
        <div class="s-meta">
          <div class="s-label">GITHUB</div>
          <div class="s-url gh-url">github.com/deshan-5</div>
        </div>
        <div class="s-arrow">›</div>
      </a>

      <a href="https://www.linkedin.com/in/deshan-gautam-66574331" target="_blank" class="social-card li float2">
        <div class="social-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="rgba(157,78,221,0.85)">
            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
          </svg>
        </div>
        <div class="s-meta">
          <div class="s-label">LINKEDIN</div>
          <div class="s-url li-url">deshan-gautam</div>
        </div>
        <div class="s-arrow">›</div>
      </a>

      <!-- Availability status -->
      <div class="avail">
        <div class="avail-dot"></div>
        <div class="avail-text">OPEN TO COLLABORATE</div>
      </div>

      <!-- Mini radar graphic -->
      <div style="margin-top:22px;text-align:center;">
        <svg width="110" height="110" viewBox="0 0 110 110" style="opacity:0.55;">
          <circle cx="55" cy="55" r="48" fill="none" stroke="rgba(0,245,255,0.12)" stroke-width="1"/>
          <circle cx="55" cy="55" r="34" fill="none" stroke="rgba(0,245,255,0.1)" stroke-width="1"/>
          <circle cx="55" cy="55" r="20" fill="none" stroke="rgba(0,245,255,0.09)" stroke-width="1"/>
          <line x1="7" y1="55" x2="103" y2="55" stroke="rgba(0,245,255,0.08)" stroke-width="1"/>
          <line x1="55" y1="7" x2="55" y2="103" stroke="rgba(0,245,255,0.08)" stroke-width="1"/>
          <polygon points="55,10 90,38 78,80 32,80 20,38" fill="rgba(0,245,255,0.04)" stroke="rgba(0,245,255,0.25)" stroke-width="1.5"/>
          <circle cx="55" cy="10" r="3" fill="#00f5ff" opacity="0.7"/>
          <circle cx="90" cy="38" r="3" fill="#9d4edd" opacity="0.7"/>
          <circle cx="78" cy="80" r="3" fill="#ff2d78" opacity="0.7"/>
          <circle cx="32" cy="80" r="3" fill="#ffb700" opacity="0.7"/>
          <circle cx="20" cy="38" r="3" fill="#00ff9d" opacity="0.7"/>
          <circle cx="55" cy="55" r="4" fill="rgba(0,245,255,0.6)"/>
          <style>
            .radar-sweep{transform-origin:55px 55px;animation:rotate 4s linear infinite;}
          </style>
          <line class="radar-sweep" x1="55" y1="55" x2="55" y2="7"
            stroke="rgba(0,245,255,0.5)" stroke-width="1.5"
            stroke-linecap="round" opacity="0.8"/>
          <path class="radar-sweep" d="M55,55 L55,7 A48,48 0 0,1 89,72 Z"
            fill="rgba(0,245,255,0.04)" stroke="none"/>
        </svg>
        <div style="font-size:7px;letter-spacing:3px;color:rgba(0,245,255,0.25);font-family:Orbitron,monospace;margin-top:2px;">SKILL RADAR</div>
      </div>
    </div>

  </div><!-- end body-grid -->

  <!-- ── COPYRIGHT BAR ── -->
  <div class="copy-bar">
    <div class="copy-left">&copy; 2025 Deshan Gautam &nbsp;&middot;&nbsp; Probability Lab &nbsp;&middot;&nbsp; All Rights Reserved</div>
    <div class="copy-right">
      <div class="copy-dot" style="background:#00f5ff;box-shadow:0 0 5px #00f5ff;"></div>
      <div class="copy-dot" style="background:#9d4edd;box-shadow:0 0 5px #9d4edd;"></div>
      <div class="copy-dot" style="background:#ff2d78;box-shadow:0 0 5px #ff2d78;"></div>
      <span class="copy-version" style="margin-left:6px;">v2.0 · 2026</span>
    </div>
  </div>

</div>
""", height=780, scrolling=False)




# PAGE: Distribution Lab.

def page_lab():
    navbar()
    lt    = st.session_state.lab_type
    dists = DISTS[lt]
    ac    = "#00f5ff" if lt == "discrete" else "#ffb700"

    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        # Back button closer to top
        st.markdown('<div style="margin-top:-8px;margin-bottom:2px;"></div>', unsafe_allow_html=True)
        if st.button("← BACK", key=f"_back_{st.session_state.page}_{st.session_state.lab_type}"): _go("landing")
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



# PAGE: HYPOTHESIS TESTING

def page_tests():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        st.markdown('<div style="margin-top:-8px;margin-bottom:2px;"></div>', unsafe_allow_html=True)
        if st.button("← BACK", key=f"_back_{st.session_state.page}_{st.session_state.lab_type}"): _go("landing")
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



# PAGE: CONFIDENCE INTERVALS

def page_ci():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        st.markdown('<div style="margin-top:-8px;margin-bottom:2px;"></div>', unsafe_allow_html=True)
        if st.button("← BACK", key=f"_back_{st.session_state.page}_{st.session_state.lab_type}"): _go("landing")
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



# PAGE: BAYES

def page_bayes():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        st.markdown('<div style="margin-top:-8px;margin-bottom:2px;"></div>', unsafe_allow_html=True)
        if st.button("← BACK", key=f"_back_{st.session_state.page}_{st.session_state.lab_type}"): _go("landing")
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



# PAGE: COMBINATORICS

def page_combinatorics():
    navbar()
    _, mid, _ = st.columns([1, 20, 1])
    with mid:
        if st.button("← BACK", key=f"_back_{st.session_state.page}_{st.session_state.lab_type}"): _go("landing")
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



# ROUTER

{
    "landing":       page_landing,
    "lab":           page_lab,
    "tests":         page_tests,
    "ci":            page_ci,
    "bayes":         page_bayes,
    "combinatorics": page_combinatorics,
}.get(st.session_state.page, page_landing)()
