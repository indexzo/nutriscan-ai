"""
Food Label Health Audit - college project
Pipeline: upload -> PIL optimise -> Gemini REST (backoff) -> JSON parse/validate
          -> deterministic scoring -> dashboard (Plotly) + PDF (fpdf2)

Run:  streamlit run app.py
Key:  set GEMINI_API_KEY as an environment variable or in .streamlit/secrets.toml
"""
import base64
import io
import json
import os
import random
import time
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import streamlit as st
from fpdf import FPDF
from PIL import Image
from pydantic import BaseModel, Field

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change if your model is retired
MAX_SIDE = 1600  # ingredient text needs more than 1200px

# ----------------------------------------------------------------------------
# 1. Data models
# ----------------------------------------------------------------------------


class Additive(BaseModel):
    name: str
    code: str = ""
    risk: str = "low"  # low | moderate | high
    effect: str = ""


class Label(BaseModel):
    legible: bool = True
    notes: str = ""
    brand: str = ""
    product_name: str = ""
    basis: str = "per 100 g"
    calories: float = 0
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    sat_fat_g: float = 0
    sugar_g: float = 0
    fiber_g: float = 0
    sodium_mg: float = 0
    ingredients: List[str] = Field(default_factory=list)
    additives: List[Additive] = Field(default_factory=list)
    contains_palm_oil: bool = False
    allergens: List[str] = Field(default_factory=list)
    dietary_tags: List[str] = Field(default_factory=list)
    child_safety: str = ""
    alternative: str = ""


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "legible": {"type": "BOOLEAN"},
        "notes": {"type": "STRING"},
        "brand": {"type": "STRING"},
        "product_name": {"type": "STRING"},
        "basis": {"type": "STRING"},
        "calories": {"type": "NUMBER"},
        "protein_g": {"type": "NUMBER"},
        "carbs_g": {"type": "NUMBER"},
        "fat_g": {"type": "NUMBER"},
        "sat_fat_g": {"type": "NUMBER"},
        "sugar_g": {"type": "NUMBER"},
        "fiber_g": {"type": "NUMBER"},
        "sodium_mg": {"type": "NUMBER"},
        "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
        "additives": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "code": {"type": "STRING"},
                    "risk": {"type": "STRING"},
                    "effect": {"type": "STRING"},
                },
                "required": ["name", "risk", "effect"],
            },
        },
        "contains_palm_oil": {"type": "BOOLEAN"},
        "allergens": {"type": "ARRAY", "items": {"type": "STRING"}},
        "dietary_tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "child_safety": {"type": "STRING"},
        "alternative": {"type": "STRING"},
    },
    "required": ["legible", "product_name", "calories", "ingredients"],
}

PROMPT = """You are an expert toxicologist and consumer health advocate.
You are given photos of one food product's packaging (front, ingredients, nutrition table).

RULES
- Transcribe ONLY what you can actually read. Never guess numbers. If a value is unreadable, use 0 and set legible=false and explain in notes.
- Give nutrition values per 100 g (convert from per-serving if needed) and state the basis in 'basis'.
- 'additives': list every E-number, preservative, colour, sweetener, emulsifier, flavour enhancer, or other concerning additive found in the ingredients. risk = low / moderate / high based on current evidence (EFSA / WHO / IARC), not fear. In 'effect', briefly explain what it does in the human body.
- Be blunt but scientifically accurate; do not call something toxic unless the evidence supports it.
- child_safety: state plainly whether it is suitable for a child daily, and why.
- alternative: a cleaner whole-food alternative.
- dietary_tags examples: Vegan, Vegetarian, Keto-friendly, Gluten-free, High-sugar, Ultra-processed.
Return JSON only, matching the schema."""

# ----------------------------------------------------------------------------
# 2. Image optimisation
# ----------------------------------------------------------------------------


def optimise(file) -> str:
    img = Image.open(file).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------
# 3. Gemini call with backoff
# ----------------------------------------------------------------------------


def call_gemini(images_b64: List[str], api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    parts = [{"text": PROMPT}] + [
        {"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in images_b64
    ]
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    last = None
    for attempt in range(5):
        r = requests.post(url, headers=headers, json=body, timeout=120)
        if r.status_code in (429, 503):  # rate limit / overloaded -> exponential backoff + jitter
            last = r.status_code
            time.sleep(2**attempt + random.random())
            continue
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(f"Gemini still busy after retries (last status {last}). Try again shortly.")


# ----------------------------------------------------------------------------
# 4. Parsing with fallback sanitiser
# ----------------------------------------------------------------------------


def parse_label(raw: str) -> Label:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return Label(**json.loads(text))


# ----------------------------------------------------------------------------
# 5. Deterministic scoring (same label -> same score every time)
# ----------------------------------------------------------------------------


def score(l: Label):
    s = 100.0
    s -= min(30, l.sugar_g * 1.5)
    s -= min(25, l.sodium_mg / 40)
    s -= min(15, l.sat_fat_g * 2)
    s -= min(10, max(0, l.calories - 250) / 30)
    risk_pts = {"low": 1, "moderate": 4, "high": 8}
    s -= min(25, sum(risk_pts.get(a.risk.lower(), 1) for a in l.additives))
    if l.contains_palm_oil:
        s -= 5
    s += min(10, l.protein_g)
    s += min(8, l.fiber_g * 2)
    s = int(max(0, min(100, round(s))))
    light = "Green" if s >= 70 else "Yellow" if s >= 40 else "Red"
    return s, light


LIGHT_COLOR = {"Green": "#2e9e4f", "Yellow": "#e0a800", "Red": "#d33a2c"}
RISK_ICON = {"low": "🟢", "moderate": "🟡", "high": "🔴"}

# ----------------------------------------------------------------------------
# 6. Charts + PDF
# ----------------------------------------------------------------------------


def macro_values(l: Label):
    return ["Protein", "Carbs", "Fat"], [l.protein_g, l.carbs_g, l.fat_g]


def plotly_donut(l: Label):
    names, vals = macro_values(l)
    fig = go.Figure(go.Pie(labels=names, values=vals, hole=0.55, textinfo="label+percent"))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, showlegend=False)
    return fig


def mpl_donut_png(l: Label) -> io.BytesIO:
    names, vals = macro_values(l)
    fig, ax = plt.subplots(figsize=(3, 3))
    if sum(vals) > 0:
        ax.pie(vals, labels=names, wedgeprops=dict(width=0.4), autopct="%1.0f%%")
    ax.set_title("Macros (g per 100 g)", fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def clean(t: str) -> str:
    """fpdf2 core fonts are latin-1 only; replace anything else."""
    return (t or "").replace("\u2019", "'").encode("latin-1", "replace").decode("latin-1")


def make_pdf(l: Label, sc: int, light: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    w = pdf.epw

    def h(text):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(235, 235, 235)
        pdf.cell(w, 8, clean(text), fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    def p(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(w, 5.5, clean(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(w, 10, "Food Label Health Audit", new_x="LMARGIN", new_y="NEXT")
    p(f"{l.brand} - {l.product_name}")

    r, g, b = [int(LIGHT_COLOR[light][i : i + 2], 16) for i in (1, 3, 5)]
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(w, 12, f"Health Score {sc}/100  |  {light}", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    h(f"Macros ({l.basis})")
    p(
        f"Calories: {l.calories:g} kcal | Protein: {l.protein_g:g} g | Carbs: {l.carbs_g:g} g | "
        f"Fat: {l.fat_g:g} g (sat. {l.sat_fat_g:g} g) | Sugar: {l.sugar_g:g} g | "
        f"Fibre: {l.fiber_g:g} g | Sodium: {l.sodium_mg:g} mg"
    )
    pdf.image(mpl_donut_png(l), x=pdf.l_margin, w=60)
    pdf.ln(2)

    h("Flagged additives")
    if l.contains_palm_oil:
        p("Palm oil: present (high in saturated fat; sustainability concerns).")
    if not l.additives:
        p("None flagged.")
    for a in l.additives:
        p(f"[{a.risk.upper()}] {a.name} {a.code}: {a.effect}")

    h("Allergens & dietary tags")
    p("Allergens: " + (", ".join(l.allergens) or "none declared"))
    p("Tags: " + (", ".join(l.dietary_tags) or "-"))

    h("Child safety")
    p(l.child_safety)
    h("Healthier alternative")
    p(l.alternative)

    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        w, 4, "Educational project output. Generated by an AI model from label photos; verify against the physical label. Not medical advice.",
        new_x="LMARGIN", new_y="NEXT",
    )
    return bytes(pdf.output())


# ----------------------------------------------------------------------------
# 7. UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Food Label Audit", page_icon="🔬", layout="wide")
st.title("🔬 Food Label Health Audit")
st.caption("Upload the front, ingredients and nutrition table photos of one product.")

api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("Set GEMINI_API_KEY (environment variable or .streamlit/secrets.toml).")
    st.stop()

files = st.file_uploader("Product photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
if files:
    st.image(files, width=160)

if st.button("Run audit", type="primary", disabled=not files):
    with st.spinner("Reading label..."):
        try:
            imgs = [optimise(f) for f in files]
            raw = call_gemini(imgs, api_key)
            try:
                st.session_state["label"] = parse_label(raw)
            except Exception:
                # one retry if the model returned broken JSON
                st.session_state["label"] = parse_label(call_gemini(imgs, api_key))
        except Exception as e:
            st.error(f"Audit failed: {e}")

label = st.session_state.get("label")
if label:
    sc, light = score(label)
    if not label.legible:
        st.warning(f"Some text was hard to read, so results may be incomplete. {label.notes}")

    top = st.columns([3, 1])
    top[0].subheader(f"{label.brand} {label.product_name}")
    top[1].markdown(
        f"<div style='background:{LIGHT_COLOR[light]};color:white;padding:14px;border-radius:12px;"
        f"text-align:center;font-size:22px;font-weight:700'>{sc}/100<br>{light}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Macros ({label.basis})**")
        m = st.columns(4)
        m[0].metric("Calories", f"{label.calories:g}")
        m[1].metric("Protein", f"{label.protein_g:g} g")
        m[2].metric("Carbs", f"{label.carbs_g:g} g")
        m[3].metric("Fat", f"{label.fat_g:g} g")
        m2 = st.columns(4)
        m2[0].metric("Sugar", f"{label.sugar_g:g} g")
        m2[1].metric("Sodium", f"{label.sodium_mg:g} mg")
        m2[2].metric("Sat. fat", f"{label.sat_fat_g:g} g")
        m2[3].metric("Fibre", f"{label.fiber_g:g} g")
    with right:
        st.plotly_chart(plotly_donut(label), use_container_width=True)

    st.markdown("### Flagged additives")
    if label.contains_palm_oil:
        st.warning("🌴 Contains palm oil: high in saturated fat; sustainability concerns.")
    if not label.additives:
        st.success("No concerning additives flagged.")
    for a in label.additives:
        with st.expander(f"{RISK_ICON.get(a.risk.lower(), '⚪')} {a.name} {a.code}  ({a.risk})"):
            st.write(a.effect)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Allergens & tags")
        st.write("**Allergens:** " + (", ".join(label.allergens) or "none declared"))
        st.write(" ".join(f"`{t}`" for t in label.dietary_tags))
        st.markdown("### Child safety")
        st.info(label.child_safety)
    with c2:
        st.markdown("### Healthier alternative")
        st.success(label.alternative)

    st.download_button(
        "⬇️ Download PDF report",
        data=make_pdf(label, sc, light),
        file_name="food_audit.pdf",
        mime="application/pdf",
    )