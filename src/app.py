import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from wrappers.openai import OpenAIWrapper
from wrappers.gemini import GeminiWrapper
from wrappers.groq import GroqWrapper
from wrappers.anthropic import AnthropicWrapper

# --- 1. ROBUST ENVIRONMENT LOADING ---
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
env_path = parent_dir / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv() 

st.set_page_config(page_title="LLM Arena", layout="wide")
st.title("⚔️ LLM Arena: Model Benchmarking")
st.markdown("Select two models, fire a prompt, and compare **Latency (TTFT)** and **Throughput** side-by-side.")

# --- 2. HELPER: SMART KEY SELECTOR ---
def render_model_selector(label_suffix, key_prefix, default_provider_index=0):
    st.subheader(f"Model {label_suffix}")
    
    provider = st.selectbox(
        f"Provider {label_suffix}", 
        ["OpenAI", "Groq", "Gemini", "Anthropic"], 
        index=default_provider_index,
        key=f"p_{key_prefix}"
    )

    env_key_map = {
        "OpenAI": "OPENAI_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY"
    }
    
    env_var_name = env_key_map.get(provider)
    system_key = os.getenv(env_var_name)
    final_api_key = None

    if system_key:
        use_demo = st.checkbox(
            "Use System Key (from .env)", 
            value=True, 
            key=f"c_{key_prefix}",
            help=f"Found {env_var_name} in your .env file."
        )
        if use_demo:
            final_api_key = system_key
            st.success(f"✅ Ready (System Key Loaded)")
        else:
            final_api_key = st.text_input(f"Enter {provider} API Key", type="password", key=f"k_{key_prefix}")
    else:
        st.warning(f"⚠️ {env_var_name} not found in .env")
        final_api_key = st.text_input(f"Enter {provider} API Key", type="password", key=f"k_{key_prefix}")

    st.divider()
    return provider, final_api_key

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🥊 Contender Selection")
    provider_a, key_a = render_model_selector("A (Left)", "a", 2) 
    provider_b, key_b = render_model_selector("B (Right)", "b", 1) 

# --- 4. FACTORY PATTERN ---
def get_wrapper(provider, key):
    if not key: return None
    try:
        if provider == "OpenAI": return OpenAIWrapper(key)
        if provider == "Groq": return GroqWrapper(key)
        if provider == "Gemini": return GeminiWrapper(key)
        if provider == "Anthropic": return AnthropicWrapper(key)
    except Exception as e:
        st.error(f"Error initializing {provider}: {e}")
    return None

model_a = get_wrapper(provider_a, key_a)
model_b = get_wrapper(provider_b, key_b)

# --- 5. PRE-CANNED PROMPTS DATA ---
prompts_category = {
    "Logic & Reasoning 🧠": [
        "How many 'r's are in the word 'strawberry'?",
        "If I have 3 apples and I eat one, then buy two more, how many do I have?",
        "If there are 3 killers in a room and I enter and kill one, how many killers are left?",
        "Which weighs more: a pound of lead or a pound of feathers?"
    ],
    "Coding & Technical 💻": [
        "Write a Python script for the Snake game using pygame.",
        "Explain the difference between TCP and UDP in 3 bullet points.",
        "Debug this: `def add(a, b): return a - b`",
        "Write a SQL query to find the second highest salary in a table."
    ],
    "Creative & Roleplay 🎭": [
        "Write a short dialogue between an angry toaster and a calm refrigerator.",
        "Explain Quantum Entanglement to a 5-year-old using emojis.",
        "Write a haiku about a robot realizing it is out of battery.",
        "Roast my choice of using Python for everything."
    ],
    "Hallucination Test 😵": [
        "Who was the President of the United States in 2035?",
        "Summarize the plot of the movie 'Gone with the Wind 2' released in 2024.",
        "What is the stock price of 'FakeCorp' right now?"
    ]
}

# --- 6. CASCADING DROPDOWN UI (UPDATED) ---
st.subheader("The Challenge")

# Initialize prompt as empty
prompt = ""

# Step 1: Category Selection (With Default Placeholder)
category_options = ["Select a Category..."] + list(prompts_category.keys())
selected_category = st.selectbox("1. Choose a Category:", category_options)

# Step 2: Question Selection (Hidden until Category is picked)
if selected_category != "Select a Category...":
    # Get questions for this category
    question_options = ["Select a Question..."] + prompts_category[selected_category]
    selected_question = st.selectbox("2. Choose a Question:", question_options)
    
    # Only set the prompt if a valid question is selected
    if selected_question != "Select a Question...":
        prompt = selected_question

# Step 3: Chat Input (Always available override)
user_input = st.chat_input("...or type a custom prompt here to override the menu")
if user_input:
    prompt = user_input

# --- 7. EXECUTION LOGIC ---
if prompt:
    if not model_a or not model_b:
        st.warning("⚠️ Please ensure API keys are available for BOTH models.")
    else:
        st.info(f"⚡ Processing Prompt: **{prompt}**")
        col1, col2 = st.columns(2)
        
        def run_column(col, model, provider_name):
            with col:
                st.markdown(f"### 🤖 {provider_name}")
                placeholder = st.empty()
                full_text = ""
                ttft = 0
                
                try:
                    gen = model.generate_stream(prompt)
                    
                    for data in gen:
                        if data["type"] == "content":
                            full_text += data["value"]
                            placeholder.markdown(full_text + "▌") 
                        elif data["type"] == "metric":
                            if data["key"] == "ttft": ttft = data["value"]
                        elif data["type"] == "final_metrics":
                            placeholder.markdown(full_text) 
                            st.success(
                                f"**TTFT:** {ttft:.4f}s | "
                                f"**Speed:** {data['tokens_per_second']:.2f} t/s"
                            )
                except Exception as e:
                    st.error(f"Error: {e}")

        run_column(col1, model_a, provider_a)
        run_column(col2, model_b, provider_b)