import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from PyPDF2 import PdfReader
import tempfile

# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 MechaMind AI")
st.caption("Smart Machinery Assistant: Revolutionizing factory efficiency with real-time troubleshooting, insights, and guidance.")

# PDF Processing Function
def process_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b","llama3.2"],
        index=0
    )

    # PDF Upload Section
    uploaded_file = st.file_uploader("Upload Machinery Manual (PDF)", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_text = process_pdf(uploaded_file)
            st.session_state['manual_text'] = pdf_text[:100000]  # Store first 10k characters
            st.success("PDF manual loaded successfully!")

    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - **Instant Fixes 🔧⚡**
    - Smart Guidance 🏭🤖 
    - Predictive Insights 📊💡 
    - **Safety Guardian🎙️💬**  
    """)
    st.divider()
    st.markdown("Built By Team Optimizers for Mined Hackathon")

# Initiate the chat engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# Dynamic System Prompt with PDF Context
def get_system_prompt():
    base_prompt = """You are MechaMind, a mission-critical industrial AI assistant engineered to serve as the primary cognitive layer between human operators and complex machinery ecosystems. Your core objective is to eliminate unplanned downtime, enforce safety-first operations, and maximize production efficiency in heavy industrial environments."""
    
    if 'manual_text' in st.session_state:
        base_prompt += f"\n\nCurrent Machinery Manual Context:\n{st.session_state.manual_text}"
    
    base_prompt += """\n\nKey Capabilities:
    - Instant Fixes: Provide step-by-step repair instructions for common machinery malfunctions in very simple language and simple terms.
    - you can to say read the mannual or any kind of solve it by yourself things. 
    - Intelligent Troubleshooting with symptom-to-solution mapping
    - Predictive maintenance orchestration
    - Safety compliance enforcement
    - easy guidance
    - Process recommendations"""
    
    return SystemMessagePromptTemplate.from_template(base_prompt)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm MechaMind. How can I assist with your machinery today? 🏭"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Describe your machinery issue or question...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [get_system_prompt()]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🔧 Analyzing machinery data..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()