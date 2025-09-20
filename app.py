import os
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# ---------------- Configuration ----------------
LLAMA_MODEL_PATH = "llama-2-7b-chat.ggmlv3.q4_0.bin"  # Put model file here
DATA_DIRECTORY = "data/"  # Folder containing your PDFs

# ---------------- Mental Health Mapping ----------------
mental_health_mapping = {
    "anxiety": {
        "possible_concerns": ["Anxiety Disorder", "Generalized Anxiety", "Panic Attacks"],
        "advice": [
            "Try deep breathing exercises like the 4-7-8 technique.",
            "Practice grounding techniques, focusing on your five senses.",
            "Challenge anxious thoughts by asking if there's solid evidence for them.",
            "Limit caffeine and sugar intake, as they can worsen anxiety.",
            "Engage in light physical activity.",
            "Consider talking to a therapist or counselor for personalized strategies."
        ],
        "resources": [
            "Information on Cognitive Behavioral Therapy (CBT)",
            "Mindfulness for anxiety guides",
            "Local mental health services"
        ]
    },
    "depression": {
        "possible_concerns": ["Depression", "Low Mood", "Persistent Sadness"],
        "advice": [
            "Try to maintain a routine, including regular sleep and mealtimes.",
            "Engage in activities you once enjoyed, even if you don't feel like it at first.",
            "Connect with supportive friends or family members.",
            "Ensure you're getting some sunlight exposure if possible.",
            "Focus on small, achievable tasks to build a sense of accomplishment.",
            "A mental health professional can offer significant support through therapy or medication."
        ],
        "resources": [
            "Information on types of depression",
            "Support groups for depression",
            "Psychotherapy options"
        ]
    },
    # ... (keep the rest of your mapping unchanged) ...
}

# ---------------- Data Loading ----------------
if not os.path.exists(DATA_DIRECTORY) or not any(fname.endswith(".pdf") for fname in os.listdir(DATA_DIRECTORY)):
    st.error(f"No PDF documents found in '{DATA_DIRECTORY}'. Please add mental health PDFs to this folder.")
    st.stop()

loader = DirectoryLoader(DATA_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

if not documents:
    st.error(f"Failed to load any documents from '{DATA_DIRECTORY}'. Ensure PDFs are valid and readable.")
    st.stop()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(text_chunks, embeddings)

# ---------------- Model Setup ----------------
# Optional: auto-download from Hugging Face (requires token + acceptance of license)
# from huggingface_hub import hf_hub_download
# if not os.path.exists(LLAMA_MODEL_PATH):
#     st.info("Downloading model from Hugging Face (this may take a while)...")
#     hf_hub_download(
#         repo_id="TheBloke/Llama-2-7B-Chat-GGML",
#         filename=LLAMA_MODEL_PATH,
#         local_dir="."
#     )

if not os.path.exists(LLAMA_MODEL_PATH):
    st.error(
        f"Llama-2 model '{LLAMA_MODEL_PATH}' not found.\n\n"
        "➡ Fix: Download 'llama-2-7b-chat.ggmlv3.q4_0.bin' from Hugging Face "
        "and place it in the project root."
    )
    st.stop()

try:
    llm = CTransformers(
        model=LLAMA_MODEL_PATH,
        model_type="llama",
        config={"max_new_tokens": 512, "temperature": 0.01, "context_length": 2048}
    )
except Exception as e:
    st.error(f"Failed to load model with ctransformers: {e}")
    st.stop()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="MindMentor: Your Mental Wellness Bot", page_icon="🧠")
st.title("🧠 MindMentor: Your Mental Wellness Bot")

st.markdown(
    """
    <div style='background-color: #ffe0b2; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <strong>Disclaimer:</strong> MindMentor is an AI assistant for informational and supportive purposes only. 
        It is NOT a substitute for professional mental health diagnosis, treatment, or advice. 
        If you are experiencing a mental health crisis, please seek immediate professional help.
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.header("Crisis Resources")
st.sidebar.markdown(
    """
    If you are in immediate danger or distress, please reach out to these resources:

    *   **National Suicide Prevention Lifeline (US):** 988  
    *   **Crisis Text Line (US):** Text HOME to 741741  
    *   **The Trevor Project (LGBTQ Youth, US):** 1-866-488-7386  
    *   **[Your Local Emergency Services]:** e.g., 911  
    """
)

# ---------------- Conversation Logic ----------------
def conversation_chat(query):
    query_lower = query.lower()
    response = ""

    # Crisis detection
    crisis_keywords = ["suicide", "harm myself", "end my life", "kill myself",
                       "hopeless", "can't go on", "want to die", "crisis", "self-harm"]
    if any(keyword in query_lower for keyword in crisis_keywords):
        return (
            "I'm really sorry to hear you're feeling this way. It sounds incredibly difficult, "
            "and I want you to know you're not alone. My purpose is to provide support, "
            "but I'm not a substitute for immediate professional help during a crisis.\n\n"
            "**Please reach out to a crisis hotline or mental health professional immediately.**\n"
            "📞 National Suicide Prevention Lifeline (US): 988\n"
            "📱 Crisis Text Line (US): Text HOME to 741741"
        )

    # Symptom mapping
    for symptom, details in mental_health_mapping.items():
        if symptom in query_lower:
            concerns = ", ".join(details["possible_concerns"])
            advice_list = details["advice"]
            response_parts = [f"It sounds like you might be experiencing {concerns}. Here's some general advice:"]
            for advice_point in advice_list:
                response_parts.append(f"• {advice_point}")
            response_parts.append("\nRemember, these are general tips — a professional can provide tailored support.")
            response = "\n".join(response_parts)
            break

    # Fallback to LLM
    if not response:
        try:
            llm_question = f"As a supportive mental wellness assistant, provide information and coping strategies for: {query}"
            result = chain({"question": llm_question, "chat_history": st.session_state['history']})
            response = result["answer"]
        except Exception as e:
            st.error(f"LLM error: {e}")
            response = "I'm having trouble processing that right now. Could you rephrase or ask something else?"

    st.session_state['history'].append((query, response))
    return response

# ---------------- Session State ----------------
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello! I'm MindMentor. How can I support you today? 🤗"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Tell me what's on your mind:",
                placeholder="E.g., I'm feeling stressed about work.",
                key="input"
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversation_chat(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

# Run app
initialize_session_state()
display_chat_history()
