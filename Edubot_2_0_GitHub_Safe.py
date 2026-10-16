import base64
import json
import re
import uuid
import asyncio
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    import edge_tts
except ImportError:
    edge_tts = None
from groq import Groq
from supabase import create_client

try:
    import extra_streamlit_components as stx
except ImportError:
    stx = None

# -------------------- APP CONFIG --------------------
st.set_page_config(
    page_title="Edubot 2.0 | AI Study Assistant",
    page_icon="🎓",
    layout="wide",
)

DEFAULT_MODEL = "openai/gpt-oss-20b"

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: linear-gradient(180deg, #f8fbff 0%, #ffffff 42%); }
    .block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
    [data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    .hero {
        padding: 1.8rem 2rem; border-radius: 24px; margin: 0 0 1.4rem 0;
        background: linear-gradient(120deg, #0f172a 0%, #312e81 55%, #2563eb 100%);
        box-shadow: 0 14px 34px rgba(37, 99, 235, .15);
        border: 1px solid rgba(255,255,255,.18);
    }
    .hero h1 { margin: 0; color: #ffffff; font-size: clamp(2rem, 4vw, 2.8rem); letter-spacing: -1.2px; }
    .hero p { margin: .65rem 0 0; color: #dbeafe; font-size: 1.02rem; line-height: 1.6; }
    .hero .eyebrow { text-transform: uppercase; letter-spacing: 2px; font-size: .72rem; font-weight: 800; color: #bfdbfe; margin-bottom: .45rem; }
    h1, h2, h3 { letter-spacing: -.35px; }
    [data-testid="stMetric"] { background: #ffffff; padding: 1rem; border: 1px solid #e2e8f0; border-radius: 16px; }
    [data-testid="stChatMessage"] { border: 1px solid #e2e8f0; border-radius: 16px; padding: .8rem 1rem; margin-bottom: .65rem; background: #ffffff; }
    [data-testid="stChatInput"] { border-radius: 16px; }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        border-radius: 12px; font-weight: 650; min-height: 2.7rem;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 5px 16px rgba(15,23,42,.10);
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
        border-radius: 11px;
    }
    [data-testid="stExpander"] { border-radius: 13px; border-color: #e2e8f0; }
    .feature-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin: .4rem 0 1.3rem; }
    .feature-card { padding: 1rem; border: 1px solid #e2e8f0; background: rgba(255,255,255,.92); border-radius: 16px; }
    .feature-card .icon { font-size: 1.45rem; }
    .feature-card strong { display: block; margin: .3rem 0; color: #0f172a; }
    .feature-card span { font-size: .85rem; color: #64748b; line-height: 1.45; }
    .small-note { font-size: .9rem; color: #64748b; }
    @media (max-width: 800px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 1.35rem; border-radius: 18px; }
        .feature-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><div class="eyebrow">Your personal learning space</div>'
    '<h1>🎓 Edubot 2.0</h1>'
    '<p>Learn smarter, one question at a time. Explore concepts, build coding skills, '
    'create revision notes, and practise with AI-powered quizzes.</p></div>',
    unsafe_allow_html=True,
)

# -------------------- SECRETS --------------------
# Keep all credentials in Streamlit secrets (or local .streamlit/secrets.toml).
# Never hard-code API keys or private configuration in this file.
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
    MODEL_NAME = st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
    VISION_MODEL = st.secrets.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    st.error(
        "Edubot configuration is missing. Add GROQ_API_KEY, SUPABASE_URL, "
        "and SUPABASE_KEY in Streamlit secrets before running the app."
    )
    st.stop()

# -------------------- SESSION STATE --------------------
DEFAULT_STATE = {
    "access_token": "",
    "refresh_token": "",
    "student_id": "",
    "student_email": "",
    "current_conversation_id": "",
    "chat_messages": [],
    "quiz_data": None,
    "quiz_topic": "",
    "quiz_id": "current",
    "last_notes": "",
    "history_load_error": "",
    "confirm_delete_chat": "",
    "saved_conversations_cache": None,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- OPTIONAL BROWSER SESSION RESTORE --------------------
# extra-streamlit-components cookies are not encrypted by this app. This is optional
# convenience only; do not treat browser-stored tokens as production-grade secure storage.
cookies = None
if stx is not None:
    try:
        cookies = stx.CookieManager(key="edubot_cookie_manager")
        if not st.session_state.access_token:
            st.session_state.access_token = cookies.get("access_token") or ""
            st.session_state.refresh_token = cookies.get("refresh_token") or ""
    except Exception:
        cookies = None

# -------------------- SUPABASE CONNECTION --------------------
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "Supabase settings are missing. Add SUPABASE_URL and SUPABASE_KEY "
        "to .streamlit/secrets.toml. Do not append /rest/v1/ to SUPABASE_URL."
    )
    st.stop()

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as exc:
    st.error("Could not connect to Supabase. Check the project URL and publishable key.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

def persist_login_cookies(access_token, refresh_token):
    """Optional convenience restore; these browser cookies are not encrypted here."""
    if cookies is not None:
        try:
            cookies.set("access_token", access_token or "", key="edubot_save_access")
            cookies.set("refresh_token", refresh_token or "", key="edubot_save_refresh")
        except Exception:
            pass


def clear_login_cookies():
    if cookies is not None:
        for cookie_name in ("access_token", "refresh_token"):
            try:
                cookies.delete(cookie_name, key=f"edubot_delete_{cookie_name}")
            except Exception:
                try:
                    cookies.set(cookie_name, "", key=f"edubot_clear_{cookie_name}")
                except Exception:
                    pass

# Restore the signed-in Supabase session.
if st.session_state.access_token and st.session_state.refresh_token:
    try:
        auth_result = supabase.auth.set_session(
            st.session_state.access_token, st.session_state.refresh_token
        )
        refreshed = getattr(auth_result, "session", None)
        if refreshed:
            st.session_state.access_token = refreshed.access_token
            st.session_state.refresh_token = refreshed.refresh_token
            persist_login_cookies(refreshed.access_token, refreshed.refresh_token)
        auth_user = getattr(supabase.auth.get_user(), "user", None)
        if auth_user:
            st.session_state.student_id = auth_user.id
            st.session_state.student_email = auth_user.email or st.session_state.student_email
        else:
            st.session_state.access_token = ""
            st.session_state.refresh_token = ""
            st.session_state.student_id = ""
    except Exception:
        st.session_state.access_token = ""
        st.session_state.refresh_token = ""
        st.session_state.student_id = ""
        st.session_state.student_email = ""
        clear_login_cookies()

# -------------------- LOGIN / SIGNUP --------------------
if not st.session_state.student_id:
    st.subheader("🔐 Student account")
    st.write("Sign in to save your chats privately and reopen them later.")
    auth_tab = st.radio("Choose an option", ["Log in", "Sign up"], horizontal=True)
    with st.form("student_auth_form", clear_on_submit=False):
        email = st.text_input("Email address", autocomplete="email")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submit_auth = st.form_submit_button(
            "Continue", type="primary", use_container_width=True
        )

    if submit_auth:
        if not email.strip() or not password:
            st.warning("Enter both your email and password.")
        elif len(password) < 6:
            st.warning("Password must be at least 6 characters. Use a strong, unique password.")
        else:
            try:
                if auth_tab == "Sign up":
                    response = supabase.auth.sign_up(
                        {"email": email.strip(), "password": password}
                    )
                    new_session = getattr(response, "session", None)
                    new_user = getattr(response, "user", None)
                    if new_session and new_user:
                        st.session_state.access_token = new_session.access_token
                        st.session_state.refresh_token = new_session.refresh_token
                        persist_login_cookies(new_session.access_token, new_session.refresh_token)
                        st.session_state.student_id = new_user.id
                        st.session_state.student_email = email.strip()
                        st.session_state.current_conversation_id = ""
                        st.session_state.chat_messages = []
                        st.success("Account created! You are signed in.")
                        st.rerun()
                    else:
                        st.success(
                            "Signup request received. If email confirmation is enabled, "
                            "verify your email, then log in."
                        )
                else:
                    response = supabase.auth.sign_in_with_password(
                        {"email": email.strip(), "password": password}
                    )
                    new_session = getattr(response, "session", None)
                    new_user = getattr(response, "user", None)
                    if not new_session or not new_user:
                        st.error("Login did not return a session. Verify your email if required.")
                    else:
                        st.session_state.access_token = new_session.access_token
                        st.session_state.refresh_token = new_session.refresh_token
                        persist_login_cookies(new_session.access_token, new_session.refresh_token)
                        st.session_state.student_id = new_user.id
                        st.session_state.student_email = new_user.email or email.strip()
                        st.session_state.current_conversation_id = ""
                        st.session_state.chat_messages = []
                        st.rerun()
            except Exception as exc:
                st.error(
                    "Authentication failed. Check your email/password, or verify your email "
                    "if signup confirmation is enabled."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))
    if stx is None:
        st.info("Optional browser session restore is unavailable because extra-streamlit-components is not installed.")
    st.caption(
        "Your password is handled by Supabase Auth; Edubot does not store it in the chat-history table."
    )
    st.stop()

# -------------------- CHAT HISTORY HELPERS --------------------
def get_saved_conversations():
    """Fetch conversation titles once per session, then reuse the cached result."""
    cached = st.session_state.get("saved_conversations_cache")
    if cached is not None:
        return cached
    try:
        result = (
            supabase.table("chat_history")
            .select("conversation_id,title,created_at")
            .eq("user_id", st.session_state.student_id)
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        summaries = {}
        for row in result.data or []:
            cid = row.get("conversation_id")
            if cid and cid not in summaries:
                summaries[cid] = {
                    "title": row.get("title") or "Untitled chat",
                    "created_at": row.get("created_at") or "",
                }
        st.session_state.saved_conversations_cache = summaries
        return summaries
    except Exception as exc:
        st.session_state.history_load_error = str(exc)
        return {}


def load_conversation(conversation_id):
    """Load only a conversation belonging to the signed-in student."""
    result = (
        supabase.table("chat_history")
        .select("role,content,conversation_id")
        .eq("user_id", st.session_state.student_id)
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    st.session_state.chat_messages = [
        {"role": row["role"], "content": row["content"]}
        for row in (result.data or [])
        if row.get("role") in ("user", "assistant") and row.get("content") is not None
    ]
    st.session_state.current_conversation_id = conversation_id


def save_chat_message(role, content):
    """Save a message and preserve an existing custom conversation title."""
    if not st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = str(uuid.uuid4())

    cid = st.session_state.current_conversation_id
    title = None
    existing = (
        supabase.table("chat_history")
        .select("title")
        .eq("user_id", st.session_state.student_id)
        .eq("conversation_id", cid)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if existing.data:
        title = existing.data[0].get("title")

    first_user_message = next(
        (
            str(item.get("content", ""))
            for item in st.session_state.chat_messages
            if item.get("role") == "user"
        ),
        "New chat",
    )
    if not title:
        title = re.sub(r"\s+", " ", first_user_message).strip()[:70] or "New chat"

    (
        supabase.table("chat_history")
        .insert(
            {
                "user_id": st.session_state.student_id,
                "conversation_id": cid,
                "role": role,
                "content": content,
                "title": title,
            }
        )
        .execute()
    )
    st.session_state.saved_conversations_cache = None


def rename_conversation(conversation_id, new_title):
    """Rename all rows in a conversation owned by the signed-in student."""
    new_title = re.sub(r"\s+", " ", new_title or "").strip()[:70]
    if not new_title:
        return False
    (
        supabase.table("chat_history")
        .update({"title": new_title})
        .eq("user_id", st.session_state.student_id)
        .eq("conversation_id", conversation_id)
        .execute()
    )
    st.session_state.saved_conversations_cache = None
    return True


def delete_conversation(conversation_id):
    """Delete a conversation and all its messages for this student."""
    (
        supabase.table("chat_history")
        .delete()
        .eq("user_id", st.session_state.student_id)
        .eq("conversation_id", conversation_id)
        .execute()
    )
    if st.session_state.current_conversation_id == conversation_id:
        st.session_state.chat_messages = []
        st.session_state.current_conversation_id = ""
    st.session_state.confirm_delete_chat = ""
    st.session_state.saved_conversations_cache = None


# -------------------- AI HELPERS --------------------
def get_client():
    if not API_KEY or API_KEY == "PASTE_YOUR_GROQ_API_KEY_HERE":
        st.error(
            "Groq API key is missing. Add GROQ_API_KEY to .streamlit/secrets.toml "
            "and keep it private."
        )
        st.stop()
    return Groq(api_key=API_KEY)


def make_system_prompt(selected_subject, selected_mode):
    prompt = (
        "You are Edubot 2.0, a patient, accurate, friendly tutor for school students. "
        "Teach for understanding, not just completion. Use simple everyday language, short paragraphs, "
        "clear headings and numbered steps when solving a problem. State the final answer clearly. "
        "For maths, write calculations on separate lines using ordinary readable characters, for example "
        "2^2 = 4 and x = sqrt(3) approximately 1.73. Do not output raw LaTeX commands such as \\frac, \\quad, \\text, "
        "or unrendered markup. Prefer plain-text maths; use $...$ only for short expressions if needed. "
        "For diagrams, describe the structure clearly or provide a simple labelled flow. "
        "Do not invent facts, sources, or citations; say when uncertain. Answer the actual question even if "
        "it differs from the selected subject. Be age-appropriate and never request passwords or API keys."
    )
    if selected_subject != "Auto-detect (Any subject)":
        prompt += f" The preferred subject is {selected_subject}."
    if selected_mode == "Simple Explanation":
        prompt += " Explain as if teaching a beginner. Define hard terms, give a relatable example, then a 2-3 point recap."
    elif selected_mode == "Coding Assistant":
        prompt += " Act as a beginner-friendly coding tutor. Give runnable code when appropriate, use fenced code blocks, explain key lines, and provide debugging steps."
    elif selected_mode == "Study Notes":
        prompt += " Create well-organized revision notes with headings, definitions, bullets, examples and a short recap."
    elif selected_mode == "AI Quiz Generator":
        prompt += " Create accurate, age-appropriate educational quizzes at the requested difficulty."
    elif selected_mode == "Diagram Maker":
        prompt += " Help plan a clear labelled educational diagram or flowchart. Use concise labels and logical connections."
    else:
        prompt += " Answer directly first, then show useful reasoning and examples."
    return prompt


def normalize_math_markup(text):
    """Convert common LaTeX delimiters to Streamlit-friendly math delimiters."""
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # Remove common spacing commands that often leak into plain-text answers.
    text = text.replace(r"\quad", " ").replace(r"\qquad", " ")
    text = text.replace(r"\,", " ").replace(r"\!", "")
    return text


def call_ai(user_prompt, system_prompt, history=None, max_tokens=1800, temperature=0.4,
            image_bytes=None, image_mime="image/jpeg", model_override=None):
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(
            [
                {"role": item["role"], "content": item["content"]}
                for item in history[-12:]
                if item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str) and item.get("content")
            ]
        )
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{image_mime};base64,{encoded}"
        user_content = [
            {"type": "text", "text": user_prompt or "Please explain this image for a student. Describe what is visible, read the question if possible, and explain step by step."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        model = model_override or VISION_MODEL
    else:
        user_content = user_prompt
        model = model_override or MODEL_NAME
    messages.append({"role": "user", "content": user_content})
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    except Exception as exc:
        # Some compatible endpoints expect max_tokens rather than max_completion_tokens.
        if "max_completion_tokens" not in str(exc).lower() and "unsupported" not in str(exc).lower():
            raise
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
    return normalize_math_markup(response.choices[0].message.content or "I couldn't create a response this time. Please try again.")


def fallback_diagram_dot(topic):
    """Local fallback so the Diagram Maker can render if the model fails."""
    t = topic.lower()
    if "amoeba" in t:
        return '''digraph Amoeba {
          graph [label="Amoeba (simplified labelled diagram)", labelloc=t, fontsize=18, rankdir=LR, bgcolor="white"];
          node [shape=box, style="rounded,filled", fillcolor="#eaf2ff", color="#2563eb", fontname="Arial"];
          amoeba [label="Amoeba cell", fillcolor="#dbeafe", penwidth=2];
          membrane [label="Cell membrane"];
          cytoplasm [label="Cytoplasm"];
          nucleus [label="Nucleus"];
          pseudopodia [label="Pseudopodia\\n(temporary extensions)"];
          vacuole [label="Food / contractile vacuole"];
          membrane -> amoeba [dir=none]; cytoplasm -> amoeba [dir=none];
          nucleus -> amoeba [dir=none]; pseudopodia -> amoeba [dir=none]; vacuole -> amoeba [dir=none];
        }'''
    safe_topic = re.sub(r'[^A-Za-z0-9 ,&()\-]', '', topic)[:70] or 'Topic'
    safe_topic = safe_topic.replace('\"', '')
    return f'''digraph EdubotDiagram {{
      graph [label="{safe_topic} - concept map", labelloc=t, fontsize=18, rankdir=TB, bgcolor="white"];
      node [shape=box, style="rounded,filled", fillcolor="#eaf2ff", color="#2563eb", fontname="Arial"];
      topic [label="{safe_topic}", fillcolor="#dbeafe", penwidth=2];
      idea1 [label="Key idea 1"]; idea2 [label="Key idea 2"]; idea3 [label="Example / application"];
      topic -> idea1; topic -> idea2; topic -> idea3;
    }}'''


def create_diagram_dot(topic, selected_subject):
    prompt = (
        "Create valid Graphviz DOT for a student-friendly educational diagram about: "
        f"{topic}. Subject: {selected_subject}. Return ONLY DOT source, no markdown fences. "
        "Use digraph G, rankdir=TB, short labels, basic shapes, and 4-10 nodes. For a biology structure such as an amoeba, label the main parts. Do not include external URLs, images, or file paths."
    )
    try:
        return call_ai(prompt, "You generate valid, safe Graphviz DOT diagrams. Return only DOT code.", max_tokens=1000, temperature=0.2)
    except Exception:
        return fallback_diagram_dot(topic)


def validate_dot(dot):
    dot = re.sub(r"^```(?:dot|graphviz)?\s*", "", dot.strip(), flags=re.IGNORECASE)
    dot = re.sub(r"\s*```$", "", dot)
    if not re.search(r"\b(digraph|graph)\b", dot) or len(dot) > 12000:
        raise ValueError("The generated diagram format was not valid. Please try another topic.")
    # Avoid HTML/script-like labels and external image/file references.
    if re.search(r"https?://|imagepath|shapefile|\bURL\s*=", dot, flags=re.IGNORECASE):
        raise ValueError("The generated diagram contained an unsupported external reference.")
    return dot

def create_quiz_prompt(topic, count, selected_subject, selected_difficulty):
    return f"""
Create exactly {count} multiple-choice quiz questions on this topic: {topic}
Selected subject: {selected_subject}
Difficulty: {selected_difficulty}

Return ONLY valid JSON. Do not use markdown fences or text outside the JSON.
Use this structure:
{{
  "title": "Short quiz title",
  "questions": [
    {{
      "question": "Question text",
      "options": ["A. first option", "B. second option", "C. third option", "D. fourth option"],
      "answer": 0,
      "explanation": "Short explanation of why the answer is correct"
    }}
  ]
}}
Rules:
- Produce exactly {count} questions.
- "answer" must be integer 0, 1, 2, or 3.
- Exactly one option is correct per question.
- Keep questions accurate, age-appropriate and suitable for the selected difficulty.
- Avoid duplicate questions.
"""


def parse_quiz(raw):
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first == -1 or last <= first:
        raise ValueError("The AI response did not contain JSON.")
    data = json.loads(cleaned[first:last + 1])
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("The quiz did not contain any questions.")
    for question in questions:
        if not isinstance(question, dict) or not isinstance(question.get("question"), str):
            raise ValueError("A question is missing its text.")
        if not isinstance(question.get("options"), list) or len(question["options"]) != 4:
            raise ValueError("Each question must have four options.")
        if not all(isinstance(option, str) and option.strip() for option in question["options"]):
            raise ValueError("Each option must contain text.")
        if not isinstance(question.get("answer"), int) or isinstance(question.get("answer"), bool) or question["answer"] not in [0, 1, 2, 3]:
            raise ValueError("A correct answer index is invalid.")
    if len(questions) > 10:
        data["questions"] = questions[:10]
    return data


def display_quiz(data, quiz_id):
    st.subheader(data.get("title", "Your AI Quiz"))
    answers = {}
    with st.form(f"quiz_form_{quiz_id}"):
        for index, question in enumerate(data["questions"]):
            st.markdown(f"**{index + 1}. {question['question']}**")
            answers[index] = st.radio(
                "Select your answer",
                question["options"],
                index=None,
                key=f"quiz_{quiz_id}_q_{index}",
                label_visibility="collapsed",
            )
        submitted = st.form_submit_button(
            "✅ Submit answers", type="primary", use_container_width=True
        )
    if submitted:
        score = 0
        st.markdown("### Your results")
        for index, question in enumerate(data["questions"]):
            correct = question["options"][question["answer"]]
            if answers[index] == correct:
                score += 1
                st.success(f"Q{index + 1}: Correct!")
            else:
                st.error(f"Q{index + 1}: Correct answer — {correct}")
            if question.get("explanation"):
                st.caption("Explanation: " + question["explanation"])
        st.metric("Final score", f"{score} / {len(data['questions'])}")
        st.progress(score / len(data["questions"]))
        percent = round(100 * score / len(data["questions"]))
        if percent == 100:
            st.balloons()
            st.success("Perfect score! Great work.")
        elif percent >= 70:
            st.success("Nice work! Review any missed answers and try again.")
        else:
            st.info("Keep practising. Read the explanations and try another quiz.")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Learning Studio")
    st.caption("Signed in: " + (st.session_state.student_email or "Student"))
    voice_enabled = st.toggle("🔊 Edubot voice", value=True, key="edubot_voice_enabled", help="Turn spoken answers on or off.")
    if not voice_enabled:
        components.html("<script>if ('speechSynthesis' in window) window.speechSynthesis.cancel();</script>", height=0)

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.current_conversation_id = str(uuid.uuid4())
        st.session_state.confirm_delete_chat = ""
        st.rerun()

    st.subheader("🗂️ My Chat History")
    saved_conversations = get_saved_conversations()
    if saved_conversations:
        for cid, info in list(saved_conversations.items())[:20]:
            label = info["title"][:38] + ("…" if len(info["title"]) > 38 else "")
            with st.container(border=True):
                if st.button(
                    "💬 " + label,
                    key="open_chat_" + cid,
                    use_container_width=True,
                ):
                    try:
                        load_conversation(cid)
                        st.session_state.confirm_delete_chat = ""
                        st.rerun()
                    except Exception as exc:
                        st.error("Could not reopen that chat. Please try again.")
                        with st.expander("Technical details"):
                            st.code(str(exc))

                with st.expander("⚙️ Chat options", expanded=False):
                    new_title = st.text_input(
                        "Chat name",
                        value=info["title"],
                        key="rename_input_" + cid,
                        max_chars=70,
                    )
                    if st.button(
                        "✏️ Rename chat",
                        key="rename_chat_" + cid,
                        use_container_width=True,
                    ):
                        try:
                            if rename_conversation(cid, new_title):
                                st.toast("Chat renamed.")
                                st.rerun()
                            else:
                                st.warning("Chat name cannot be empty.")
                        except Exception as exc:
                            st.error("Could not rename this chat.")
                            with st.expander("Technical details"):
                                st.code(str(exc))

                    if st.button(
                        "🗑️ Delete chat",
                        key="delete_chat_" + cid,
                        use_container_width=True,
                    ):
                        st.session_state.confirm_delete_chat = cid
                        st.rerun()

                    if st.session_state.confirm_delete_chat == cid:
                        st.warning("Delete this chat and all its messages?")
                        yes_col, cancel_col = st.columns(2)
                        with yes_col:
                            if st.button(
                                "Yes, delete",
                                key="confirm_delete_" + cid,
                                type="primary",
                                use_container_width=True,
                            ):
                                try:
                                    delete_conversation(cid)
                                    st.toast("Chat deleted.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error("Could not delete this chat.")
                                    with st.expander("Technical details"):
                                        st.code(str(exc))
                        with cancel_col:
                            if st.button(
                                "Cancel",
                                key="cancel_delete_" + cid,
                                use_container_width=True,
                            ):
                                st.session_state.confirm_delete_chat = ""
                                st.rerun()
    else:
        st.caption("Your saved chats will appear here.")
    if st.button("🚪 Log out", use_container_width=True):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        clear_login_cookies()
        for key in [
            "access_token", "refresh_token", "student_id", "student_email",
            "current_conversation_id", "chat_messages", "confirm_delete_chat",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.caption("Choose how you want to study.")
    subject = st.selectbox(
        "📚 Subject (optional)",
        [
            "Auto-detect (Any subject)", "General", "Science", "Mathematics",
            "English", "Computer Science", "Social Science", "History",
            "Geography", "Economics", "Law / CLAT",
        ],
        index=0,
    )
    mode = st.radio(
        "🎯 Learning mode",
        ["AI Chat Tutor", "Simple Explanation", "AI Quiz Generator", "Coding Assistant", "Study Notes", "Diagram Maker"],
    )
    st.divider()
    difficulty = st.select_slider(
        "Quiz difficulty", options=["Easy", "Medium", "Hard"], value="Medium"
    )
    question_count = st.slider("Quiz questions", min_value=3, max_value=10, value=5)
    st.caption("Edubot is a learning assistant. Verify important facts with textbooks/teachers.")

# -------------------- MAIN FEATURES --------------------
if mode == "Diagram Maker":
    st.subheader("🧩 Diagram Maker")
    st.caption("Create a simple labelled flowchart or concept map from a topic. This creates a diagram, not a generated photo.")
    diagram_topic = st.text_input("What should the diagram explain?", placeholder="e.g., Photosynthesis, water cycle, how a bill becomes law")
    if st.button("✨ Create diagram", type="primary", use_container_width=True):
        if not diagram_topic.strip():
            st.warning("Enter a topic first.")
        else:
            with st.spinner("Edubot is building your diagram..."):
                try:
                    try:
                        dot = validate_dot(create_diagram_dot(diagram_topic.strip(), subject))
                    except Exception:
                        dot = validate_dot(fallback_diagram_dot(diagram_topic.strip()))
                    st.graphviz_chart(dot, use_container_width=True)
                    if "amoeba" in diagram_topic.lower():
                        st.caption("Simplified school-level diagram. Check labels against your textbook.")
                    st.download_button("⬇️ Download diagram source (.dot)", data=dot, file_name="edubot_diagram.dot", mime="text/vnd.graphviz")
                except Exception as exc:
                    st.error("The diagram could not be rendered. Please check Streamlit's Graphviz renderer.")
                    with st.expander("Technical details"):
                        st.code(str(exc))

elif mode == "AI Quiz Generator":
    st.subheader("📝 AI Quiz Generator")
    st.write("Create a multiple-choice quiz on almost any topic.")
    topic = st.text_input(
        "Enter a topic", placeholder="e.g., Photosynthesis, Fractions, Fundamental Rights"
    )
    if st.button("✨ Generate quiz", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Please enter a topic first.")
        else:
            with st.spinner("Edubot is creating your quiz..."):
                try:
                    raw = call_ai(
                        create_quiz_prompt(topic.strip(), question_count, subject, difficulty),
                        "You are an accurate quiz creator. Follow the requested JSON structure exactly.",
                        max_tokens=3500,
                        temperature=0.2,
                    )
                    st.session_state.quiz_data = parse_quiz(raw)
                    st.session_state.quiz_topic = topic.strip()
                    st.session_state.quiz_id = str(uuid.uuid4())[:12]
                except Exception as exc:
                    st.error("Quiz creation failed. Check the API key, model, account limits, and connection.")
                    with st.expander("Technical details"):
                        st.code(str(exc))
    if st.session_state.quiz_data:
        try:
            display_quiz(st.session_state.quiz_data, st.session_state.quiz_id)
        except Exception as exc:
            st.error("The quiz could not be displayed. Generate it again.")
            with st.expander("Technical details"):
                st.code(str(exc))

elif mode == "Study Notes":
    st.subheader("📒 Study Notes Maker")
    topic = st.text_input("Topic for revision notes", placeholder="e.g., The Indian Constitution")
    note_style = st.selectbox(
        "Notes style", ["Quick revision", "Detailed notes", "Exam questions + answers"]
    )
    if st.button("Create study notes", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Making your notes..."):
                try:
                    prompt = (
                        f"Create {note_style.lower()} on '{topic.strip()}' for the subject {subject}. "
                        "Use clear headings and student-friendly language. Do not invent citations. "
                        "If useful, include definitions, examples, key points and a short recap."
                    )
                    st.session_state.last_notes = call_ai(
                        prompt, make_system_prompt(subject, "Study Notes"), max_tokens=2600
                    )
                except Exception as exc:
                    st.error("Could not create notes. Check the API connection and model settings.")
                    with st.expander("Technical details"):
                        st.code(str(exc))
    if st.session_state.last_notes:
        st.markdown(st.session_state.last_notes)
        st.download_button(
            "⬇️ Download notes (.md)",
            data=st.session_state.last_notes,
            file_name="edubot_study_notes.md",
            mime="text/markdown",
        )

else:
    heading = {
        "AI Chat Tutor": "💬 Ask Edubot",
        "Simple Explanation": "💡 Simple Explanation Mode",
        "Coding Assistant": "💻 Coding Assistant",
    }[mode]
    description = {
        "AI Chat Tutor": "Ask questions about any topic.",
        "Simple Explanation": "Get beginner-friendly explanations, examples, and a short recap.",
        "Coding Assistant": "Learn programming, understand code, or ask for debugging help.",
    }[mode]
    st.subheader(heading)
    st.caption(description)
    # Attachments and voice recording are built into Streamlit's chat input below.
    # This keeps the controls inside the message bar instead of in a separate panel.
    if "voice_transcribed_text" not in st.session_state:
        st.session_state.voice_transcribed_text = ""

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_value = st.chat_input(
        "Message Edubot…",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"],
        accept_audio=True,
        key="edubot_chat_input",
    )

    user_text = ""
    image_bytes = None
    image_mime = "image/jpeg"

    if prompt_value is not None:
        user_text = (getattr(prompt_value, "text", "") or "").strip()
        attached_files = getattr(prompt_value, "files", []) or []
        audio_attachment = getattr(prompt_value, "audio", None)

        if attached_files:
            image_file = attached_files[0]
            image_bytes = image_file.getvalue()
            image_mime = getattr(image_file, "type", None) or "image/jpeg"
            if not user_text:
                user_text = "Please explain this image for a student. Read any visible question and explain it step by step."
            if len(attached_files) > 1:
                st.info("For now, Edubot processes one picture per message; the first picture will be used.")

        if audio_attachment is not None:
            try:
                audio_type = getattr(audio_attachment, "type", None) or "audio/wav"
                extension = "webm" if "webm" in audio_type else ("mp3" if "mpeg" in audio_type else "wav")
                with st.spinner("Transcribing your voice..."):
                    transcription = get_client().audio.transcriptions.create(
                        file=(f"voice_question.{extension}", audio_attachment.getvalue(), audio_type),
                        model="whisper-large-v3-turbo",
                        response_format="text",
                    )
                transcribed = str(transcription).strip()
                if transcribed:
                    user_text = f"{user_text} {transcribed}".strip()
                    st.success("Voice recognized.")
                elif not user_text:
                    st.warning("I couldn't hear clear words. Please record again or type your question.")
            except Exception as exc:
                st.error("Voice transcription failed. Check your Groq API access or type your question instead.")
                with st.expander("Technical details"):
                    st.code(str(exc))

    if user_text:
        st.session_state.voice_transcribed_text = ""
        display_user_text = user_text + ("\n\n📷 Image attached" if image_bytes else "")
        if not st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = str(uuid.uuid4())

        st.session_state.chat_messages.append({"role": "user", "content": display_user_text})
        with st.chat_message("user"):
            st.markdown(display_user_text)

        try:
            save_chat_message("user", user_text)
        except Exception as exc:
            st.warning("Your message could not be saved to history, but Edubot will still try to answer.")
            with st.expander("Save error details"):
                st.code(str(exc))

        with st.chat_message("assistant"):
            with st.spinner("Edubot is thinking..."):
                try:
                    system = make_system_prompt(subject, mode)
                    history = st.session_state.chat_messages[:-1]
                    answer = call_ai(
                        user_text, system, history=history,
                        image_bytes=image_bytes, image_mime=image_mime,
                    )
                    st.markdown(answer)
                    # Natural neural voice (Indian English) for a friendly teacher-like delivery.
                    if st.session_state.get("edubot_voice_enabled", True):
                        answer_id = str(uuid.uuid5(uuid.NAMESPACE_URL, answer))
                        if st.session_state.get("edubot_last_spoken_answer_id") != answer_id:
                            st.session_state.edubot_last_spoken_answer_id = answer_id
                            if edge_tts is None:
                                st.info("For a more natural teacher-like voice, install Edge TTS: pip install edge-tts")
                            else:
                                # Remove Markdown syntax and keep speech clear and conversational.
                                speech_text = re.sub(r"```.*?```", " Code omitted. ", answer, flags=re.S)
                                speech_text = re.sub(r"!\[.*?\]\(.*?\)", " Image. ", speech_text)
                                speech_text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", speech_text)
                                speech_text = re.sub(r"[#*_`>]+", "", speech_text)
                                speech_text = re.sub(r"\s+", " ", speech_text).strip()
                                try:
                                    async def _make_teacher_voice(text, output_path):
                                        communicate = edge_tts.Communicate(
                                            text,
                                            voice="en-IN-NeerjaNeural",
                                            rate="-5%",
                                            pitch="+2Hz",
                                            volume="+0%",
                                        )
                                        await communicate.save(output_path)

                                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as voice_file:
                                        voice_path = voice_file.name
                                    asyncio.run(_make_teacher_voice(speech_text, voice_path))
                                    with open(voice_path, "rb") as audio_file:
                                        st.audio(audio_file.read(), format="audio/mp3", autoplay=True)
                                    try:
                                        Path(voice_path).unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                except Exception:
                                    st.caption("Voice playback couldn't start automatically. Use the audio player to listen.")
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    try:
                        save_chat_message("assistant", answer)
                    except Exception as save_exc:
                        st.warning("Edubot answered, but the reply could not be saved to history.")
                        with st.expander("Save error details"):
                            st.code(str(save_exc))
                except Exception as exc:
                    st.error(
                        "Edubot could not respond. Check the Groq API key, model, connection, and account limits."
                    )
                    with st.expander("Technical details"):
                        st.code(str(exc))

st.divider()
st.caption("Edubot 2.0 • Learn step by step. Verify important facts with trusted learning materials.")
