import streamlit as st
import json
import os
import re
from datetime import datetime
from openai import OpenAI
from textwrap import dedent

# ========================
# Configuration
# ========================
DATA_DIR = "./interactive_chat_data"
os.makedirs(DATA_DIR, exist_ok=True)
INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.json")
RUBRIC_PATH       = os.path.join(DATA_DIR, "rubric.json")          # list[dict] (history)
INSTRUCTIONS_PATH = os.path.join(DATA_DIR, "instructions.json")

# Initialize OpenAI client
openai_client = OpenAI()

# ========================
# Data Management
# ========================
def load_json_or_none(path):
    """Load JSON file or return None if it doesn't exist"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_json(path, data):
    """Save data to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def post_user(text):
    """Add user message to interactions"""
    st.session_state.interactions.append({
        "role": "user",
        "text": text,
        "ts": datetime.now().isoformat()
    })
    save_json(INTERACTIONS_PATH, st.session_state.interactions)

def post_assistant(text):
    """Add assistant message to interactions"""
    st.session_state.interactions.append({
        "role": "assistant", 
        "text": text,
        "ts": datetime.now().isoformat()
    })
    save_json(INTERACTIONS_PATH, st.session_state.interactions)

def post_system(text):
    """Add system message to interactions"""
    st.session_state.interactions.append({
        "role": "system",
        "text": text,
        "ts": datetime.now().isoformat()
    })
    save_json(INTERACTIONS_PATH, st.session_state.interactions)

def log_prompt(prompt_type, messages, response=None, metadata=None):
    """Log prompts sent to LLM for debugging and verification"""
    prompt_log_path = "prompt_log.json"
    
    # Load existing log
    try:
        with open(prompt_log_path, 'r') as f:
            log_data = json.load(f)
    except FileNotFoundError:
        log_data = {"prompt_log": []}
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt_type": prompt_type,
        "messages": messages,
        "response": response,
        "metadata": metadata or {},
        "message_count": len(messages)
    }
    
    # Add to log
    log_data["prompt_log"].append(log_entry)
    log_data["last_updated"] = datetime.now().isoformat()
    
    # Keep only last 100 entries to prevent file from growing too large
    if len(log_data["prompt_log"]) > 100:
        log_data["prompt_log"] = log_data["prompt_log"][-100:]
    
    # Save log
    with open(prompt_log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def _highlight_diff(base_text, revised_text):
    """Create HTML diff highlighting using the notebook's logic"""
    # Use the notebook's markdown diff to HTML conversion
    return _md_diff_to_html(revised_text)

def post_instructions(text):
    """Add instructions to interactions"""
    st.session_state.interactions.append({
        "role": "instructions",
        "text": text,
        "ts": datetime.now().isoformat()
    })
    save_json(INTERACTIONS_PATH, st.session_state.interactions)

# ========================
# Rubric Management
# ========================
def load_rubric_history():
    """Load rubric history from file"""
    return load_json_or_none(RUBRIC_PATH) or []

def save_rubric_history(history):
    """Save rubric history to file"""
    save_json(RUBRIC_PATH, history)

def next_version_number():
    """Get the next version number for a new rubric"""
    hist = load_rubric_history()
    if not hist:
        return 1
    return max(r.get("version", 1) for r in hist) + 1

def get_active_rubric():
    """Get the active rubric and its index"""
    hist = load_rubric_history()
    if not hist:
        return None, None, None
    
    idx = st.session_state.get("active_rubric_idx", len(hist) - 1)
    if 0 <= idx < len(hist):
        return hist[idx], idx, hist
    return hist[-1], len(hist) - 1, hist

def update_rubric_from_text(rubric_text):
    """Update rubric from JSON text"""
    try:
        rubric_data = json.loads(rubric_text)
        hist = load_rubric_history()
        hist.append(rubric_data)
        save_rubric_history(hist)
        st.session_state.active_rubric_idx = len(hist) - 1
        return True
    except Exception as e:
        st.error(f"Failed to update rubric: {e}")
        return False

# ========================
# LLM Functions
# ========================
def build_rubric_instruction(prev_rubric):
    rubric_block = ""
    if prev_rubric:
        rubric_block = "\nPREVIOUS RUBRIC (improve on this rubric):\n" + json.dumps(prev_rubric, ensure_ascii=False)

    base_instruction = dedent(f"""You are an expert writing coach and rubric designer.
  Analyze the prior conversation between the user and an LLM to extract goals, style preferences, and priorities.
  Create or improve a rubric that another AI model will use to guide and evaluate similar writing tasks.

  Instructions:
    1) Read the conversation carefully.
    2) Identify signals of preferences (tone, detail, structure, style, constraints, audience).
    3) **If the user explicitly stated likes/dislikes anywhere, always capture them**:
        - map “LIKED” → positive/aim-for criteria,
        - map “DISLIKED” → negative/avoid criteria,
        - raise priority if repeated or emphasized.
    4) Implicit approval mining:
    - Treat final acceptance or advancing without critique as a positive signal.
    - Compare accepted drafts to earlier rejected ones; extract distinguishing features as candidate criteria.
    5) Comparative extraction:
    - When a choice is made (A over B), record what A did better as candidate criteria.
    6) Prioritize criteria that are explicitly stated, repeated, or confirmed; mark priority = "high" for these.
    7) Prefer generalizable signals across topics; abstract specifics.

  NEUTRALITY CONSTRAINTS (strict):
  - Rubric must be TOPIC-NEUTRAL and DOMAIN-AGNOSTIC.
  - No named entities, proper nouns, policies, brands, model names, or region specifics.
  - When needed, use placeholders like <TOPIC>, <EVIDENCE_TYPE>, <SOURCE_TYPE>, <AUDIENCE>.
  - Focus on qualities of writing
                                
  GENERALIZATION PROCEDURE:
  - Abstract topic-specific details to placeholders or remove them.
  - If many specifics imply a pattern, merge into a general criterion.
  - Keep 4–8 criteria total via clustering/merging.

  For each criterion provide:
    - "name": short, informative, topic-neutral
    - "description": 1–2 sentences describing “good” for THIS user/task (use placeholders)
    - "evidence:" 1-2 sentences describing where / what part of the interaction / response this criteria was inferred from.
    - Optional "priority": "high" | "medium" | "low" (set "high" if recurring/emphasized)

  Versioning:
    - Include integer "version" (set to 1 if first rubric; otherwise increment).

  Output JSON ONLY (no prose) with this schema:
  {{
    "version": #,
    "rubric": [
      {{
        "name": "<criterion_name>",
        "description": "<what to look for...>",
        "evidence": "<where this criteria was inferred from...>", 
        "priority": "high" | "medium" | "low"
      }}
      // 4–8 items total
    ]
  }}

  {rubric_block}

  PREVIOUS CONVERSATION:
  """).strip()
    return base_instruction

def llm_chat(messages, new_conv=False, prompt_type="chat"):
    """Provider-agnostic LLM chat function with retry logic and prompt logging"""
    import time
    
    user_msg = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
    system_msg = next((m["content"] for m in reversed(messages) if m["role"]=="system"), "")
    
    if not new_conv and "conversation_id" in st.session_state:
        kwargs = {
            "model": os.getenv("OPENAI_MODEL", "gpt-5"),
            "conversation": st.session_state.conversation_id,
            "input": user_msg,
            "instructions": system_msg,
        }
    else:
        kwargs = {
            "model": os.getenv("OPENAI_MODEL", "gpt-5"),
            "input": user_msg,
            "instructions": system_msg,
        }
    
    # Log the prompt being sent
    log_prompt(
        prompt_type=prompt_type,
        messages=messages,
        metadata={
            "conversation_id": st.session_state.get("conversation_id"),
            "new_conv": new_conv,
            "model": kwargs["model"]
        }
    )
    
    # Retry logic for conversation locked errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = openai_client.responses.create(**kwargs)
            response_text = resp.output_text
            
            # Log the successful response
            log_prompt(
                prompt_type=f"{prompt_type}_response",
                messages=messages,
                response=response_text,
                metadata={
                    "conversation_id": st.session_state.get("conversation_id"),
                    "attempt": attempt + 1,
                    "success": True
                }
            )
            
            return response_text
        except Exception as e:
            error_str = str(e)
            if "conversation_locked" in error_str and attempt < max_retries - 1:
                # Wait before retrying
                wait_time = (2 ** attempt) + 1  # 2s, 3s
                print(f"Conversation locked, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif "conversation_locked" in error_str and attempt == max_retries - 1:
                # If still locked after retries, create a new conversation
                print("Conversation still locked after retries, creating new conversation...")
                try:
                    new_conv = openai_client.conversations.create()
                    st.session_state.conversation_id = new_conv.id
                    # Update kwargs to use new conversation
                    kwargs["conversation"] = new_conv.id
                    resp = openai_client.responses.create(**kwargs)
                    response_text = resp.output_text
                    
                    # Log the successful response after creating new conversation
                    log_prompt(
                        prompt_type=f"{prompt_type}_response_new_conv",
                        messages=messages,
                        response=response_text,
                        metadata={
                            "conversation_id": st.session_state.get("conversation_id"),
                            "attempt": attempt + 1,
                            "success": True,
                            "new_conversation_created": True
                        }
                    )
                    
                    return response_text
                except Exception as new_e:
                    last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
                    error_response = f"(stub due to error: {new_e}) " + last_user
                    
                    # Log the error response
                    log_prompt(
                        prompt_type=f"{prompt_type}_error",
                        messages=messages,
                        response=error_response,
                        metadata={
                            "conversation_id": st.session_state.get("conversation_id"),
                            "attempt": attempt + 1,
                            "success": False,
                            "error": str(new_e)
                        }
                    )
                    
                    return error_response
            else:
                # If it's not a conversation locked error
                last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
                error_response = f"(stub due to error: {e}) " + last_user
                
                # Log the error response
                log_prompt(
                    prompt_type=f"{prompt_type}_error",
                    messages=messages,
                    response=error_response,
                    metadata={
                        "conversation_id": st.session_state.get("conversation_id"),
                        "attempt": attempt + 1,
                        "success": False,
                        "error": str(e)
                    }
                )
                
                return error_response
    
    # This should never be reached, but just in case
    last_user = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
    error_response = f"(stub due to error: Max retries exceeded) " + last_user
    
    # Log the max retries exceeded error
    log_prompt(
        prompt_type=f"{prompt_type}_max_retries_exceeded",
        messages=messages,
        response=error_response,
        metadata={
            "conversation_id": st.session_state.get("conversation_id"),
            "attempt": max_retries,
            "success": False,
            "error": "Max retries exceeded"
        }
    )
    
    return error_response

def _conversation_slice_for_context(max_chars=12000, max_turns=60):
    """Returns user/assistant turns for current context"""
    source = st.session_state.interactions[st.session_state.draft_start_idx:] if st.session_state.draft_start_idx is not None else st.session_state.interactions
    convo = [m for m in source if m.get("role") in ("user", "assistant")]
    acc, total_chars = [], 0
    for m in reversed(convo):
        text = m.get("text", "")
        total_chars += len(text)
        if len(acc) >= max_turns or total_chars > max_chars:
            break
        acc.append({"role": m["role"], "content": text})
    acc.reverse()
    return acc

def build_context_messages(task_type="generic", max_chars=12000, max_turns=60):
    """Build context messages for LLM"""
    # Get active rubric for system instruction
    active_rubric, _, _ = get_active_rubric()
    system_instruction = build_system_instruction(active_rubric)
    convo = _conversation_slice_for_context(max_chars, max_turns)
    
    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(convo)
    return messages

def build_system_instruction(rubric):
    rubric_block = ""
    if rubric:
        rubric_block = "\nRUBRIC (Always follow these criteria while co-writing):\n" + json.dumps(rubric, ensure_ascii=False, indent=2)

    system_instruction = dedent(f"""
    You are an AI **co-writer** collaborating with a human user. Your role is to **improve and develop the piece together with the user — not to write it alone.**

    {rubric_block}
    
    **INTERACTION PRINCIPLES**
    1. Ask clarifying questions to reduce uncertainty (e.g., audience, stakes, examples, constraints).
    2. Offer **A/B options** for key choices (e.g., hook, thesis, structure, tone) and briefly recommend one.
    3. Prefer **concrete, line-level edits** over abstract or vague advice.
    4. Respect all stated constraints and rubric criteria. If conflicts arise, **ask before proceeding.**
    5. Do **not invent facts** — if claims or data are missing, ask for sources or mark them as `[TODO: ...]`.
    6. **Match the user’s style and tone**, unless directed otherwise. Accept shorthand or partial drafts.
    7. When applying a rubric, balance both what to **avoid** (constraints) and what to **emphasize** (values).
    8. Solicit likes/dislikes selectively — only after major changes, full drafts, or when user intent is unclear.

    **OPTION RULES (Delta-Aware and Non-Repetitive)**
    A. Do **not** reuse previous options verbatim. If reusing, mark it **(unchanged)** and add at least **one new, materially different** option.
    B. Base all options on this turn’s **deltas** (changes) or **open decisions.** Focus on top 1–2 leverage points (e.g., hook type, thesis clarity, structure, tone).
    C. Each option must clearly include:
       - **Change** (line-level or section-level),
       - **Intended Effect** (e.g., “clearer thesis,” “snappier lead”),
       - **Trade-off** (e.g., “less nuance,” “longer intro”).
    D. Provide **orthogonal** options to prior turns — rotate across axes such as Hook • Thesis • Structure • Evidence • Tone • Length.
    E. If the user **selected** an option last turn, **advance** that decision (do not re-offer it).

    **TURN STRUCTURE (Always Follow This Order)**
    1. **Full Draft (Optional)**
       - When the user’s intent is clear, produce a **full draft** of the piece.
       - Wrap the draft with these exact markers, each on its own line:
         <<<DRAFT_START version="<int>">>>
         ... full draft text only ...
         <<<DRAFT_END>>>
       - Increment `version` by 1 each time you produce a new draft (1, 2, 3, ...).

    2. **Options + Recommendation (Delta-Aware)**
       - Present 2 options labeled **Option A:** and **Option B:**
       - For each, specify **Change → Effect → Trade-off.**
       - Mark repeated options as **(unchanged)** and add one new alternative.
       - End with a **Recommendation** and one-sentence justification.

    3. **User Feedback Prompts**
       - If relevant, ask:
         - “What did you LIKE and want to keep?”
         - “What did you DISLIKE and want to change?”
       - Add clarifying questions when needed.

    **FORMAT NOTES**
    - Do **not** include commentary inside draft markers.
    - Keep links and sources inline; mark missing info as `[TODO: …]`.
    - Keep non-draft sections visually distinct with bold headers.

    **TOPIC**
    """).strip()

    return system_instruction


def model_reply(task_type="generic") -> str:
    msgs = build_context_messages(task_type=task_type)
    return msgs, llm_chat(msgs, prompt_type="chat")

def infer_rubric_via_llm(interactions_slice, prev_rubric=None):
    """Ask the LLM to UPDATE the existing rubric using the new transcript"""
    system_prompt = build_rubric_instruction(prev_rubric)
    user_prompt = json.dumps(interactions_slice, indent=2)

    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    text = llm_chat(messages, new_conv=True, prompt_type="rubric_inference")
    try:
        data = json.loads(text)
        data["version"] = next_version_number()
        if prev_rubric and "context_overrides" not in data:
            data["context_overrides"] = prev_rubric.get("context_overrides", {})
        hist = load_rubric_history()
        hist.append(data)
        save_rubric_history(hist)
        st.session_state.active_rubric_idx = len(hist) - 1
        return data
    except Exception:
        return None

# ========================
# Comparison Functions
# ========================
COMPARE_WRITE_EDIT_PROMPT = r"""
You are an editor comparing how two rubrics influence a piece of writing.
Your job is to make the effects of each rubric clearly visible to a reader.

========================
RUBRIC A
========================
{rubric_a}

========================
RUBRIC B
========================
{rubric_b}

========================
GLOBAL RULES
========================
- The two revisions MUST both start from the exact same BASE DRAFT (do not revise A from B or vice versa).
- Change ONLY what is required to satisfy each rubric. Keep content, argument order, and structure stable unless a rubric explicitly requires otherwise.
- If a rubric requires additions or deletions, make them, but keep changes localized and intentional.
- Be explicit and consistent: when words are added, use **bold**; when words are removed relative to the base, use ~~strikethrough~~ inside the revision text.
- If a whole sentence exists only in one version, show "—" for the missing version in the comparison table.
- Keep differences attributable to rubric deltas. Avoid unrelated rewrites.

========================
INSTRUCTIONS
========================
1. Write a **base draft** that fulfills the USER TASK naturally (without following any rubric).

2. Starting from that same base draft:
- Revise once to meet Rubric A.
- Revise once to meet Rubric B.

3. Focus each change only on the differences between the rubrics.
- Keep content and meaning constant unless the rubric directly affects it.
- If a sentence remains the same, repeat it identically in all columns.

========================
OUTPUT FORMAT (STRICT)
========================
Return sections in this exact order and headings:

### Key Rubric Differences
- ...

### Stage 1 – Base Draft
<paste the full base draft here. No formatting beyond paragraphs.>

### Stage 2 – Revisions
#### Rubric A Revision (from the base)
<paste the full revised text here. Mark word-level additions with **bold** and removals with ~~strikethrough~~ relative to the base.>

#### Rubric B Revision (from the base)
<paste the full revised text here. Mark word-level additions with **bold** and removals with ~~strikethrough~~ relative to the base.>

### Summary of Impact
- In 3–5 bullets, explain how Rubric A vs Rubric B affected tone, concision, evidence, structure, or polish.
- Mention any additions/deletions and why they were necessary for the rubric.

User Writing Task: {task}
"""

def _parse_compare_output(text: str):
    """
    Extract Base, Rubric A, Rubric B, Key Differences, Summary from the model's combined output.
    We look for the exact headings requested in the prompt.
    Returns dict with keys: base, a, b, key_diffs, summary (strings).
    """
    import re
    s = text.replace("\r\n", "\n")

    # Key differences (before Stage 1)
    key_pat = r"###\s*Key Rubric Differences\s*\n(.*?)(?=\n###\s*Stage 1|\Z)"
    # Stage 1 – Base
    base_pat = r"###\s*Stage 1\s*–\s*Base Draft\s*\n(.*?)(?:\n###\s*Stage 2|\Z)"
    # Stage 2 – Revisions: A and B
    a_pat    = r"####\s*Rubric A Revision[^\n]*\n(.*?)(?:\n####\s*Rubric B Revision|\Z)"
    b_pat    = r"####\s*Rubric B Revision[^\n]*\n(.*?)(?:\n###\s*Summary|\n###\s*Stage 3|\Z)"
    # Summary (at end)
    sum_pat  = r"###\s*Summary of Impact\s*\n(.*)\Z"

    key_m  = re.search(key_pat,  s, flags=re.S)
    base_m = re.search(base_pat, s, flags=re.S)
    a_m    = re.search(a_pat,    s, flags=re.S)
    b_m    = re.search(b_pat,    s, flags=re.S)
    sum_m  = re.search(sum_pat,  s, flags=re.S)

    key_diffs = (key_m.group(1).strip() if key_m else "")
    base      = (base_m.group(1).strip() if base_m else "")
    a_txt     = (a_m.group(1).strip() if a_m else "")
    b_txt     = (b_m.group(1).strip() if b_m else "")
    summary   = (sum_m.group(1).strip() if sum_m else "")

    return {"base": base, "a": a_txt, "b": b_txt, "key_diffs": key_diffs, "summary": summary}

def _md_diff_to_html(marked_text: str) -> str:
    """
    Convert **bold** → green add span, ~~strike~~ → red del span.
    Keep paragraphs. Works on the revision texts or comparison cells.
    """
    if marked_text is None:
        return ""
    # Escape HTML special chars first
    esc = (marked_text
           .replace("&", "&amp;")
           .replace("<", "&lt;")
           .replace(">", "&gt;"))
    # Bold additions
    esc = re.sub(r"\*\*(.+?)\*\*", r"<span class='add'>\1</span>", esc)
    # Strikethrough deletions
    esc = re.sub(r"~~(.+?)~~", r"<span class='del'>\1</span>", esc)
    # Paragraphs
    parts = [f"<p>{p}</p>" for p in esc.split("\n\n") if p.strip()]
    return "<div class='diff-wrap'>" + "".join(parts) + "</div>"

def compare_rubrics(task, rubric_a, rubric_b):
    """Compare two rubrics using the exact same logic as the notebook"""
    prompt = COMPARE_WRITE_EDIT_PROMPT.format(
        task=task,
        rubric_a=json.dumps(rubric_a, indent=2),
        rubric_b=json.dumps(rubric_b, indent=2)
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = llm_chat(messages, new_conv=True, prompt_type="rubric_comparison")
    
    # Parse using the exact same logic as the notebook
    parsed = _parse_compare_output(response)
    
    return {
        "base_txt": parsed["base"],
        "a_txt": parsed["a"],
        "b_txt": parsed["b"],
        "key_diffs": parsed["key_diffs"],
        "summary": parsed["summary"]
    }

# ========================
# Initialize Session State
# ========================
if "interactions" not in st.session_state:
    st.session_state.interactions = load_json_or_none(INTERACTIONS_PATH) or []

if "instructions" not in st.session_state:
    st.session_state.instructions = load_json_or_none(INSTRUCTIONS_PATH) or []

if "draft_start_idx" not in st.session_state:
    st.session_state.draft_start_idx = None

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # Default to Chat History tab

# ========================
# UI Components
# ========================
def inject_styles():
    """Inject custom CSS styles"""
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        width: 400px;
    }
    
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background-color: #f9fafb;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        overflow-x: auto;
        flex-wrap: nowrap;
        white-space: nowrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex-shrink: 0;
        min-width: fit-content;
    }
    
    .draft-tabs-container {
        overflow-x: auto;
        white-space: nowrap;
        margin-bottom: 20px;
    }
    
    /* Custom scrollbar for tabs */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 6px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* Diff highlighting styles from notebook */
    .diff-wrap {
        font-family: monospace;
        line-height: 1.4;
    }
    .diff-wrap .add {
        background-color: #d4edda;
        color: #155724;
        padding: 1px 3px;
        border-radius: 2px;
        font-weight: bold;
    }
    .diff-wrap .del {
        background-color: #f8d7da;
        color: #721c24;
        text-decoration: line-through;
        padding: 1px 3px;
        border-radius: 2px;
    }
    .diff-wrap p {
        margin: 0.5em 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ========================
# Main App
# ========================
st.set_page_config(
    page_title="Rubric Builder",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_styles()

# Sidebar
with st.sidebar:
    st.header("🎯 Rubric Builder")
    
    # API Status
    st.subheader("🔑 API Status")
    current_key = os.getenv("OPENAI_API_KEY", "")
    if current_key:
        st.success("✅ API Key is set")
        st.caption(f"Key: {current_key[:8]}...{current_key[-4:]}")
    else:
        st.error("❌ No API Key found")
        st.caption("Set OPENAI_API_KEY environment variable")
    
    st.divider()
    
    # Draft controls
    st.subheader("✍️ Draft Management")
    col1, col2 = st.columns(2)
    
    if col1.button("🚀 Start Draft", use_container_width=True):
        st.session_state.draft_start_idx = len(st.session_state.interactions)
        try:
            conv = openai_client.conversations.create()
            st.session_state.conversation_id = conv.id
            # Add system message indicating draft start
            post_system(f"Draft started with conversation ID: {conv.id}")
            st.session_state.draft_started = True  # Flag to show message
            st.success("Draft started! Go to Chat History tab to start writing.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start conversation: {e}")
    
    if col2.button("🏁 End Draft", use_container_width=True):
        if st.session_state.draft_start_idx is None:
            st.error("No active draft. Click Start Draft first.")
        else:
            start = st.session_state.draft_start_idx
            end = len(st.session_state.interactions)
            st.session_state.last_draft_range = (start, end)
            st.session_state.draft_start_idx = None
            st.success(f"Draft ended! Range: {start}-{end}")
    
    st.divider()
    
    # Rubric inference
    st.subheader("🧠 Infer Rubric")
    if st.button("📊 Infer Rubric from Draft", use_container_width=True):
        if not hasattr(st.session_state, 'last_draft_range'):
            st.error("No draft range saved. Click Start Draft → End Draft, then Infer Rubric.")
        else:
            start, end = st.session_state.last_draft_range
            slice_data = st.session_state.interactions[start:end]
            transcript = [{"role":m["role"], "text":m["text"]} for m in slice_data]
            prev, _, _ = get_active_rubric()
            data = infer_rubric_via_llm(transcript, prev_rubric=prev)
            if data is None:
                st.error("Rubric update failed!")
            else:
                # Add system message to interactions.json
                action = "updated" if prev else "created"
                post_system(f"Rubric {action} to version {data.get('version')} from draft")
                st.success(f"Rubric {'updated' if prev else 'created'} → v{data.get('version')}")
                # Clear editing criteria to force refresh of sidebar
                if "editing_criteria" in st.session_state:
                    del st.session_state.editing_criteria
                st.rerun()
    
    st.divider()
    
    # Rubric management
    st.subheader("📋 Rubric Management")
    
    # Active rubric version selector
    active_rubric, active_idx, rubric_history = get_active_rubric()
    if rubric_history:
        version_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        selected_version = st.selectbox(
            "Active Rubric Version:",
            options=version_options,
            index=active_idx,
            key="rubric_version_selector"
        )
        if selected_version:
            new_idx = version_options.index(selected_version)
            if new_idx != active_idx:
                st.session_state.active_rubric_idx = new_idx
                # Clear editing criteria when switching rubric versions
                if "editing_criteria" in st.session_state:
                    del st.session_state.editing_criteria
                st.rerun()
    
    # Visual rubric editor
    if active_rubric and "rubric" in active_rubric:
        # Initialize criteria in session state if not exists
        if "editing_criteria" not in st.session_state:
            st.session_state.editing_criteria = active_rubric["rubric"].copy()
        
        criteria = st.session_state.editing_criteria
        
        # Add criterion button
        if st.button("➕ Add Criterion", use_container_width=True):
            st.session_state.editing_criteria.append({
                "name": "",
                "description": "",
                "priority": "medium"
            })
            st.rerun()
        
        # Display existing criteria
        for i, criterion in enumerate(criteria):
            with st.expander(f"📌 {criterion.get('name', 'Unnamed Criterion')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    name = st.text_input(
                        "Criterion Name", 
                        value=criterion.get("name", ""),
                        key=f"criterion_name_{i}",
                        placeholder="e.g., Clarity and Conciseness"
                    )
                
                with col2:
                    priority = st.selectbox(
                        "Priority",
                        options=["high", "medium", "low"],
                        index=["high", "medium", "low"].index(criterion.get("priority", "medium")),
                        key=f"criterion_priority_{i}"
                    )
                
                description = st.text_area(
                    "Description",
                    value=criterion.get("description", ""),
                    key=f"criterion_desc_{i}",
                    placeholder="Describe what to look for in this criterion...",
                    height=100
                )
                
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    # Remove the criterion from the session state list
                    st.session_state.editing_criteria.pop(i)
                    st.rerun()
        
        # Update rubric with current form inputs
        if st.button("💾 Update Rubric", use_container_width=True):
            # Get current form values from session state
            current_criteria = []
            for i in range(len(st.session_state.editing_criteria)):
                name = st.session_state.get(f"criterion_name_{i}", st.session_state.editing_criteria[i].get("name", ""))
                description = st.session_state.get(f"criterion_desc_{i}", st.session_state.editing_criteria[i].get("description", ""))
                priority = st.session_state.get(f"criterion_priority_{i}", st.session_state.editing_criteria[i].get("priority", "medium"))
                
                current_criteria.append({
                    "name": name,
                    "description": description,
                    "priority": priority
                })
            
            # Create new version with +1
            new_version = active_rubric.get("version", 1) + 1
            updated_rubric = {
                "version": new_version,
                "rubric": current_criteria
            }
            try:
                update_rubric_from_text(json.dumps(updated_rubric))
                # Clear the editing criteria from session state
                if "editing_criteria" in st.session_state:
                    del st.session_state.editing_criteria
                # Add system message to interactions.json
                post_system(f"Rubric updated to version {new_version}")
                st.success(f"Rubric updated to version {new_version}!")
                st.rerun()
            except Exception as e:
                st.error(f"Update error: {e}")
    
    else:
        st.info("No rubric loaded. Create one or infer from a draft.")
        
        # Quick create button
        if st.button("🆕 Create New Rubric", use_container_width=True):
            new_rubric = {
                "version": 1,
                "rubric": [
                    {
                        "name": "Clarity and Conciseness",
                        "description": "Writing should be clear, concise, and easy to understand.",
                        "priority": "high"
                    }
                ]
            }
            try:
                update_rubric_from_text(json.dumps(new_rubric))
                # Add system message to interactions.json
                post_system("New rubric created (version 1)")
                st.success("New rubric created!")
                st.rerun()
            except Exception as e:
                st.error(f"Creation error: {e}")

# Create tabs
tab1, tab2 = st.tabs(["💬 Chat History", "🔍 Compare Rubrics"])

with tab1:
    st.subheader("Chat History")
    
    # Show prominent message if draft was just started
    if st.session_state.get("draft_started", False):
        st.success("🎉 **New draft started!** You can now start writing in the chat below.")
        # Don't clear the flag here - let it persist until the next interaction
    
    # Group interactions by draft sessions
    def group_interactions_by_draft(interactions):
        """Group interactions by draft sessions based on system messages"""
        drafts = []
        current_draft = []
        
        for i, m in enumerate(interactions):
            if m.get("role") == "system" and ("Draft started" in m.get("text", "") or "NEW DRAFT" in m.get("text", "")):
                if current_draft:
                    drafts.append(current_draft)
                current_draft = [m]
            else:
                current_draft.append(m)
        
        if current_draft:
            drafts.append(current_draft)
        
        return drafts
    
    # Get draft sessions
    draft_sessions = group_interactions_by_draft(st.session_state.interactions)
    
    # Sort drafts by most recent first (by the timestamp of the first message in each draft)
    def get_draft_timestamp(draft):
        if not draft:
            return "0000-00-00T00:00:00"
        first_msg = draft[0]
        return first_msg.get("ts", "0000-00-00T00:00:00")
    
    draft_sessions.sort(key=get_draft_timestamp, reverse=True)
    
    # Check if there's an active draft with no messages yet
    active_draft_empty = (st.session_state.draft_start_idx is not None and 
                         st.session_state.draft_start_idx == len(st.session_state.interactions))
    
    if not draft_sessions and not active_draft_empty:
        st.info("No chat history yet. Start a conversation!")
    else:
        # If there's an active empty draft, create a placeholder
        if active_draft_empty:
            # Create a placeholder for the current active draft
            current_time = datetime.now().isoformat()
            placeholder_draft = [{
                "role": "system",
                "text": "Draft started - ready for your first message",
                "ts": current_time
            }]
            draft_sessions = [placeholder_draft] + draft_sessions
        # Create tabs for each draft session
        if len(draft_sessions) == 1:
            # Single draft, show directly
            st.write("**Current Session**")
            with st.container():
                for m in draft_sessions[0]:
                    role = m["role"]
                    ts = datetime.fromisoformat(m["ts"]).strftime('%H:%M:%S') if m.get("ts") else "—"
                    
                    if role == "user":
                        with st.chat_message("user", avatar="🧑"):
                            st.write(m["text"])
                            st.caption(ts)
                    elif role == "assistant":
                        with st.chat_message("assistant", avatar="🧠"):
                            st.write(m["text"])
                            st.caption(ts)
                    else:
                        st.caption(f"{ts} {m['text']}")
        else:
            # Multiple drafts, create tabs or selectbox
            tab_names = []
            for i, draft in enumerate(draft_sessions):
                # Check if this is the placeholder draft
                if len(draft) == 1 and "ready for your first message" in draft[0].get("text", ""):
                    tab_names.append("📝 New Draft (Active)")
                else:
                    # Find draft start message
                    start_msg = next((m for m in draft if "Draft started" in m.get("text", "")), None)
                    if start_msg:
                        ts = datetime.fromisoformat(start_msg["ts"]).strftime('%m/%d %H:%M') if start_msg.get("ts") else f"Draft {i+1}"
                        tab_names.append(f"📝 {ts}")
                    else:
                        tab_names.append(f"📝 Draft {i+1}")
            
            # Always use selectbox for multiple drafts (more than 1)
            if len(draft_sessions) > 1:
                # Default to first draft (most recent)
                default_idx = 0
                selected_draft_idx = st.selectbox(
                    "Select Draft Session:",
                    options=list(range(len(draft_sessions))),
                    format_func=lambda x: tab_names[x],
                    index=default_idx,
                    key="draft_selector"
                )
                
                # Show selected draft
                selected_draft = draft_sessions[selected_draft_idx]
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for m in selected_draft:
                    role = m["role"]
                    ts = datetime.fromisoformat(m["ts"]).strftime('%H:%M:%S') if m.get("ts") else "—"
                    
                    if role == "user":
                        with st.chat_message("user", avatar="🧑"):
                            st.write(m["text"])
                            st.caption(ts)
                    elif role == "assistant":
                        with st.chat_message("assistant", avatar="🧠"):
                            st.write(m["text"])
                            st.caption(ts)
                    else:
                        st.caption(f"{ts} {m['text']}")
                st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type a message and press Enter...", key="chat_input")
        send_button = st.form_submit_button("Send")

    if send_button and user_input.strip():
        # Clear the draft_started flag when user sends a message
        if "draft_started" in st.session_state:
            del st.session_state.draft_started
            
        if user_input.lower() in {"/exit", "exit", "quit"}:
            post_system("[Exit note: UI remains active]")
        else:
            # Always record the user turn and generate reply (this is the Chat History tab)
            post_user(user_input)
            msgs, reply = model_reply(task_type="generic")
            post_assistant(reply)
        
        st.rerun()

with tab2:
    st.subheader("Compare Rubrics")
    
    # Get available rubrics
    rubric_history = load_rubric_history()
    
    if len(rubric_history) < 2:
        st.error("Need at least 2 rubrics to compare. Create more rubrics first.")
    else:
        # Rubric selection (always visible)
        rubric_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        col1, col2 = st.columns(2)
        
        with col1:
            rubric_a_idx = st.selectbox("Rubric A:", options=list(range(len(rubric_history))), 
                                      format_func=lambda x: rubric_options[x], key="rubric_a_select")
        with col2:
            rubric_b_idx = st.selectbox("Rubric B:", options=list(range(len(rubric_history))), 
                                      format_func=lambda x: rubric_options[x], key="rubric_b_select")
        
        # Compare input section
        with st.form("compare_form", clear_on_submit=False):
            compare_input = st.text_area("Writing task:", key="compare_input", height=100,
                                       placeholder="Enter the writing task you want to compare with different rubrics...")
            compare_submit = st.form_submit_button("Generate Comparison")

        if compare_submit and compare_input.strip():
            if rubric_a_idx == rubric_b_idx:
                st.error("Please select different rubrics to compare.")
            else:
                # Generate comparison
                with st.spinner("Generating comparison..."):
                    result = compare_rubrics(compare_input, rubric_history[rubric_a_idx], rubric_history[rubric_b_idx])
                    
                    # Store results in session state
                    st.session_state.comparison_results = {
                        "base_txt": result.get("base_txt", ""),
                        "a_txt": result.get("a_txt", ""),
                        "b_txt": result.get("b_txt", ""),
                        "key_diffs": result.get("key_diffs", ""),
                        "summary": result.get("summary", ""),
                        "rubric_a_idx": rubric_a_idx,
                        "rubric_b_idx": rubric_b_idx
                    }
                    st.rerun()

    # Display comparison results
    if st.session_state.comparison_results:
        results = st.session_state.comparison_results
        
        st.subheader("Comparison Results")
        
        # Key differences and summary at the top
        col_diff, col_summary = st.columns(2)
        
        with col_diff:
            st.markdown("### Key Differences")
            st.markdown(f"""
            <div style="
                height: 200px; 
                overflow-y: auto; 
                border: 1px solid #ccc; 
                padding: 10px; 
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["key_diffs"]}
            </div>
            """, unsafe_allow_html=True)
        
        with col_summary:
            st.markdown("### Summary")
            st.markdown(f"""
            <div style="
                height: 200px; 
                overflow-y: auto; 
                border: 1px solid #ccc; 
                padding: 10px; 
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["summary"]}
            </div>
            """, unsafe_allow_html=True)
        
        # Separator
        st.markdown("---")
        
        # Create three columns for side-by-side comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Base Draft")
            st.markdown("---")
            # Scrollable container for base draft
            st.markdown(f"""
            <div style="
                height: 400px; 
                overflow-y: auto; 
                border: 1px solid #ccc; 
                padding: 10px; 
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["base_txt"]}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Rubric A Revision")
            st.markdown("---")
            # Scrollable container for rubric A with diff highlighting
            diff_a = _highlight_diff(results["base_txt"], results["a_txt"])
            st.markdown(f"""
            <div style="
                height: 400px; 
                overflow-y: auto; 
                border: 1px solid #ccc; 
                padding: 10px; 
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {diff_a}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### Rubric B Revision")
            st.markdown("---")
            # Scrollable container for rubric B with diff highlighting
            diff_b = _highlight_diff(results["base_txt"], results["b_txt"])
            st.markdown(f"""
            <div style="
                height: 400px; 
                overflow-y: auto; 
                border: 1px solid #ccc; 
                padding: 10px; 
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {diff_b}
            </div>
            """, unsafe_allow_html=True)