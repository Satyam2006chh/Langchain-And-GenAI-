import streamlit as st
from dotenv import load_dotenv
import os
import time
from typing import List, Dict
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

@st.cache_resource(show_spinner=False)
def create_chat_model(repo_id: str, task: str = "text-generation"):
    """
    Create and cache the Hugging Face endpoint and Chat wrapper.
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("❌ HUGGINGFACEHUB_API_TOKEN not found in .env file.")

    llm = HuggingFaceEndpoint(repo_id=repo_id, task=task)
    chat = ChatHuggingFace(llm=llm)
    return chat



st.set_page_config(page_title="🤖 LangChain Chatbot", layout="centered")
st.title("🤖 SmartBot")

st.markdown(
    """
    ### 💬 How to use:
    - Type your question below and hit **Send**.  
    - Click **Show History** to see the full conversation.  
    - Click **Clear History** to start fresh.  
    - Ensure your `.env` file contains:  
      ```
      HUGGINGFACEHUB_API_TOKEN=hf_xxx_your_token
      ```
    """
)

MODEL_OPTIONS = {
    "Mistral (7B Instruct)": "mistralai/Mistral-7B-Instruct-v0.2",
    "Zephyr (7B Chat)": "HuggingFaceH4/zephyr-7b-alpha",
    "Llama 2 (7B Chat)": "meta-llama/Llama-2-7b-chat-hf",
    "Flan-T5 (Large)": "google/flan-t5-large"
}

col1, col2 = st.columns([3, 1])
with col1:
    selected_label = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()), index=0)
    repo_id = MODEL_OPTIONS[selected_label]
with col2:
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "You are a helpful AI assistant that gives clear and concise answers."}
    ]

if "show_history" not in st.session_state:
    st.session_state.show_history = False

try:
    chat_model = create_chat_model(repo_id=repo_id)
except Exception as e:
    st.error(f"Model creation failed: {e}")
    st.stop()

user_input = st.text_input("Your message:", placeholder="Type your question here...")

col_send, col_show, col_clear = st.columns([1, 1, 1])
with col_send:
    send = st.button("Send")
with col_show:
    if st.button("Show History"):
        st.session_state.show_history = not st.session_state.show_history
with col_clear:
    if st.button("Clear History"):
        st.session_state.history = [
            {"role": "system", "content": "You are a helpful AI assistant that gives clear and concise answers."}
        ]
        st.success("Chat history cleared!")

if st.session_state.show_history:
    st.markdown("### 📜 Conversation History")
    for msg in st.session_state.history:
        role = msg["role"]
        if role == "system":
            continue
        elif role == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        elif role == "ai":
            st.markdown(f"**🤖 AI:** {msg['content']}")
    st.markdown("---")

if send and user_input.strip() != "":
    # Append user message
    st.session_state.history.append({"role": "user", "content": user_input.strip()})

    
    messages = []
    for m in st.session_state.history:
        if m["role"] == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    # Placeholder for progressive text
    placeholder = st.empty()
    status = st.empty()

    try:
        status.info("🤖 Thinking...")

        result = chat_model.invoke(messages)

        ai_text = ""
        if hasattr(result, "content"):
            ai_text = result.content
        elif isinstance(result, dict):
            ai_text = result.get("text") or next(iter(result.values()), "")
        else:
            ai_text = str(result)

       
        streamed = ""
        for char in ai_text:
            streamed += char
            placeholder.markdown(f"**🤖 AI:** {streamed}▌")
            time.sleep(0.01)
        placeholder.markdown(f"**🤖 AI:** {ai_text}")

        # Save to history
        st.session_state.history.append({"role": "ai", "content": ai_text})
        status.success("✅ Response completed")

    except Exception as err:
        st.error(f"❌ Error while generating: {err}")
        st.stop()

    # Clear input after send
    st.session_state.user_input = ""

# -------------------------
st.markdown("### 💬 Latest Conversation")
for msg in st.session_state.history[-10:]:
    if msg["role"] == "system":
        continue
    elif msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    elif msg["role"] == "ai":
        st.markdown(f"**🤖 AI:** {msg['content']}")

st.markdown("---")
st.caption("Built with ❤️ By Satyam chhabra")




