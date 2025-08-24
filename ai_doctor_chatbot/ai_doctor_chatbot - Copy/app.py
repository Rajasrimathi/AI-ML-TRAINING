# app.py
import os
import random
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------- Page Setup ---------------------------
st.set_page_config(
    page_title="AI Doctor Assistant",
    page_icon="🩺",
    layout="wide"
)

# Minimal theming & CSS
st.markdown("""
<style>
/* Tighter layout & nicer chat bubbles */
.main .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
.chat-bubble {
  padding: .9rem 1rem; border-radius: 12px; margin: .25rem 0 .6rem 0; line-height: 1.55;
  box-shadow: 0 1px 2px rgba(0,0,0,.06);
}
.user { background: rgba(0, 122, 255, .08); border: 1px solid rgba(0, 122, 255, .20); }
.assistant { background: rgba(16, 185, 129, .08); border: 1px solid rgba(16, 185, 129, .20); }
.small { font-size: 0.88rem; opacity: .9; }
.kbd {
  background: #111; color: #fff; border-radius: 6px; padding: 2px 6px; font-size: .8rem;
}
.badge {
  display:inline-block; padding:4px 8px; border-radius: 999px; font-size: .78rem;
  border:1px solid rgba(0,0,0,.1); background: rgba(255,255,255,.5);
}
</style>
""", unsafe_allow_html=True)

# --------------------------- Sidebar Controls ---------------------------
st.sidebar.title("⚙️ Settings")

# Model path: allow user override (default = fine_tuned_model in the same folder)
default_model_dir = "./fine_tuned_model"
model_dir = st.sidebar.text_input("Model path", value=default_model_dir, help="Path to a saved Hugging Face model dir")

# Generation controls
st.sidebar.subheader("🧠 Generation")
max_new_tokens = st.sidebar.slider("Max new tokens", 16, 256, 80, step=8)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.8, step=0.05)
top_p = st.sidebar.slider("Top-p (nucleus)", 0.1, 1.0, 0.95, step=0.05)
top_k = st.sidebar.slider("Top-k", 0, 200, 50, step=10, help="0 disables top-k sampling")
repetition_penalty = st.sidebar.slider("Repetition penalty", 1.0, 2.0, 1.05, step=0.05)
seed_on = st.sidebar.checkbox("Use deterministic seed", value=False)
seed_val = st.sidebar.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1, help="Only used if 'Use deterministic seed' is ON")

# Appearance
st.sidebar.subheader("🎨 Appearance")
hero_url = st.sidebar.text_input("Header image URL (optional)")
show_examples = st.sidebar.checkbox("Show quick symptom buttons", value=True)

# Session tools
colA, colB = st.sidebar.columns(2)
with colA:
    clear_btn = st.button("🧹 Clear chat")
with colB:
    reload_btn = st.button("🔁 Reload model")

# --------------------------- Load Model ---------------------------
@st.cache_resource(show_spinner=True)
def load_model_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(path)
    mdl.eval()
    return mdl, tok

# initial load
_model, _tokenizer = load_model_tokenizer(model_dir)

# allow manual reload
if reload_btn:
    # Clear the cache for this function and reload
    load_model_tokenizer.clear()
    _model, _tokenizer = load_model_tokenizer(model_dir)
    st.sidebar.success("Model reloaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(device)

# --------------------------- Helpers ---------------------------
def system_greeting():
    greetings = [
        "👋 Hi! I’m your AI Doctor Assistant. Tell me your symptoms, and I’ll share general advice.",
        "🩺 Hello! Describe what you’re feeling. I’ll respond with general guidance.",
        "😊 Welcome! Enter your symptoms below for supportive suggestions."
    ]
    return random.choice(greetings)

def build_prompt(symptom_text: str) -> str:
    return f"### Patient:\nI have {symptom_text}.\n\n### Doctor:\n"

@torch.inference_mode()
def generate_response(symptom_text: str) -> str:
    if seed_on:
        torch.manual_seed(int(seed_val))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed_val))

    prompt = build_prompt(symptom_text)
    inputs = _tokenizer(prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.pad_token_id
    )
    if top_k > 0:
        gen_kwargs["top_k"] = int(top_k)

    out = _model.generate(**inputs, **gen_kwargs)
    text = _tokenizer.decode(out[0], skip_special_tokens=True)

    # Return only the doctor's part if possible
    # Split on "### Doctor:\n" and drop the prompt prefix
    if "### Doctor:" in text:
        try:
            doctor_part = text.split("### Doctor:")[1].strip()
            return doctor_part
        except Exception:
            pass
    return text

# --------------------------- Session State ---------------------------
if "messages" not in st.session_state or clear_btn:
    st.session_state.messages = [
        {"role": "assistant", "content": system_greeting()}
    ]

# --------------------------- Header ---------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### 🩺 AI Doctor Assistant")
    st.markdown('<span class="badge">Demo • Not medical advice</span>', unsafe_allow_html=True)
with col2:
    if hero_url.strip():
        st.image(hero_url, use_container_width=True)

st.markdown(
    '<div class="small">💡 Tip: Try symptoms like <span class="kbd">fever</span>, '
    '<span class="kbd">sore throat</span>, <span class="kbd">headache</span>, or '
    '<span class="kbd">stomach pain</span>.</div>',
    unsafe_allow_html=True
)

# --------------------------- Quick Examples ---------------------------
example_symptoms = [
    "fever and body ache",
    "sore throat and dry cough",
    "headache with light sensitivity",
    "stomach pain and nausea",
    "shortness of breath on exertion",
    "skin rash and itching",
    "dizziness and fatigue",
]

if show_examples:
    st.write("**🔎 Quick symptoms:**")
    ex_cols = st.columns(7)
    for i, s in enumerate(example_symptoms):
        if ex_cols[i].button(f"📝 {s}"):
            st.session_state.messages.append({"role": "user", "content": s})

# --------------------------- Chat Display ---------------------------
for msg in st.session_state.messages:
    avatar = "🧑‍⚕️" if msg["role"] == "assistant" else "🧑"
    bubble_class = "assistant" if msg["role"] == "assistant" else "user"
    with st.container():
        st.markdown(
            f'<div class="chat-bubble {bubble_class}"><strong>{avatar} '
            f'{"Doctor" if msg["role"]=="assistant" else "You"}:</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )

# --------------------------- Input (Chat) ---------------------------
user_input = st.chat_input("Describe your symptoms…")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate answer
    with st.spinner("Thinking…"):
        try:
            reply = generate_response(user_input)
        except Exception as e:
            reply = f"⚠️ I ran into an error while generating a response: `{e}`"

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Re-render the new two messages immediately
    avatar_user = "🧑"
    st.markdown(
        f'<div class="chat-bubble user"><strong>{avatar_user} You:</strong><br>{user_input}</div>',
        unsafe_allow_html=True
    )
    avatar_doc = "🧑‍⚕️"
    st.markdown(
        f'<div class="chat-bubble assistant"><strong>{avatar_doc} Doctor:</strong><br>{reply}</div>',
        unsafe_allow_html=True
    )

# --------------------------- Footer / Disclaimers ---------------------------
st.markdown("---")
st.caption(
    "⚠️ **Important:** This AI provides general, educational guidance only and may be inaccurate. "
    "It is **not** a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider with any questions about a medical condition."
)

# Small device / compute indicator
st.sidebar.markdown("---")
st.sidebar.write(
    "💻 **Device:** " + ("CUDA GPU" if torch.cuda.is_available() else "CPU") +
    "  •  📂 **Model dir:** " + os.path.abspath(model_dir)
)

# footer
