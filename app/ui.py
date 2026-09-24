import html

import streamlit as st

from traffic.classes import CLASS_COLORS_HEX, CLASS_LABELS_VI, CLASS_NAMES
from traffic.config import load_config, resolve
from traffic.storage import TrafficDB

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {
  background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 55%, #0EA5E9 100%);
  color: #fff; border-radius: 16px; padding: 28px 32px; margin-bottom: 1.2rem;
}
.hero h1 {color: #fff; margin: 0 0 6px 0; font-size: 1.9rem; padding: 0;}
.hero p {color: #DBEAFE; margin: 0; font-size: 1rem;}
.kpi {
  background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 14px 18px;
  border-left: 6px solid var(--accent); box-shadow: 0 1px 2px rgba(15,23,42,.04); margin-bottom: 12px;
}
.kpi.sm {padding: 10px 12px; border-left-width: 4px;}
.kpi.sm .label {font-size: .72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.kpi.sm .value {font-size: 1.5rem;}
.kpi.sm .sub {font-size: .72rem;}
.kpi .label {color: #64748B; font-size: .82rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {color: #0F172A; font-size: 2rem; font-weight: 700; line-height: 1.2; font-variant-numeric: tabular-nums;}
.kpi .sub {color: #64748B; font-size: .8rem;}
.step {
  background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; height: 100%;
  text-align: center;
}
.step .n {display:inline-block; width:28px; height:28px; line-height:28px; border-radius:50%;
  background:#2563EB; color:#fff; font-weight:700; font-size:.85rem; margin-bottom:6px;}
.step .t {font-weight: 700; color: #0F172A;}
.step .d {color: #64748B; font-size: .82rem;}
.feed {font-size: .88rem; line-height: 1.9;}
.pill {display:inline-block; padding: 1px 9px; border-radius: 999px; color:#fff; font-size:.75rem; font-weight:600;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def kpi(label: str, value, accent: str = "#2563EB", sub: str = "", small: bool = False):
    st.markdown(
        f'<div class="kpi{" sm" if small else ""}" style="--accent:{accent}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div><div class="sub">{html.escape(sub)}&nbsp;</div></div>',
        unsafe_allow_html=True,
    )


def class_kpis(by_class: dict, total: int, extra_sub: dict | None = None):
    cols = st.columns(len(CLASS_NAMES) + 1)
    with cols[0]:
        kpi("Tổng phương tiện", f"{total:,}", "#0F172A", (extra_sub or {}).get("total", ""))
    for col, cls in zip(cols[1:], CLASS_NAMES):
        n = by_class.get(cls, 0)
        pct = f"{n / total:.0%} tổng" if total else ""
        with col:
            kpi(CLASS_LABELS_VI[cls], f"{n:,}", CLASS_COLORS_HEX[cls], pct)


def pill(cls: str) -> str:
    return f'<span class="pill" style="background:{CLASS_COLORS_HEX[cls]}">{CLASS_LABELS_VI[cls]}</span>'


@st.cache_resource
def get_db() -> TrafficDB:
    return TrafficDB(resolve(load_config()["storage"]["db_path"]))
