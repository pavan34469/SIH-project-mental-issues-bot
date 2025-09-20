import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import os

# --- Configuration ---
# Path to your Llama-2 model. Make sure this file is in your project directory
# or provide the full path to it.
# You can download it from Hugging Face: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
LLAMA_MODEL_PATH = "llama-2-7b-chat.ggmlv3.q4_0.bin"
DATA_DIRECTORY = 'data/'

# --- 1. Mental Health Symptom/Keyword Mapping ---
# This dictionary provides specific, curated responses for common mental health terms.
# It acts as a "first line of defense" for common queries, ensuring consistent and helpful advice.
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
    "stress": {
        "possible_concerns": ["Chronic Stress", "Burnout", "Overwhelm"],
        "advice": [
            "Identify your stressors and try to manage or reduce them.",
            "Practice relaxation techniques like progressive muscle relaxation.",
            "Ensure you're getting adequate sleep.",
            "Incorporate regular physical activity into your day.",
            "Set realistic boundaries in your work and personal life.",
            "Consider delegating tasks if possible.",
            "Learning to say 'no' can be a powerful stress reducer."
        ],
        "resources": [
            "Stress management techniques",
            "Work-life balance tips",
            "Mindfulness for stress reduction"
        ]
    },
    "insomnia": {
        "possible_concerns": ["Sleep Disturbances", "Difficulty Falling/Staying Asleep"],
        "advice": [
            "Establish a consistent sleep schedule, even on weekends.",
            "Create a relaxing bedtime routine (e.g., warm bath, reading).",
            "Make sure your bedroom is dark, quiet, and cool.",
            "Avoid caffeine and heavy meals close to bedtime.",
            "Limit screen time an hour before bed.",
            "If worries keep you up, try writing them down earlier in the evening."
        ],
        "resources": [
            "Sleep hygiene guidelines",
            "Relaxation techniques for sleep",
            "Information on CBT for insomnia"
        ]
    },
    "sadness": {
        "possible_concerns": ["Low Mood", "Temporary Sadness"],
        "advice": [
            "Allow yourself to feel the emotion without judgment.",
            "Reach out to a friend or loved one to share how you're feeling.",
            "Engage in a comforting activity.",
            "Listen to uplifting music or watch a favorite movie.",
            "Practice self-compassion.",
            "If sadness persists or becomes overwhelming, seeking professional guidance is a good step."
        ],
        "resources": [
            "Emotional regulation strategies",
            "Self-compassion exercises"
        ]
    },
    "anger": {
        "possible_concerns": ["Anger Management Issues", "Irritability"],
        "advice": [
            "Take a deep breath and count to ten before reacting.",
            "Identify the triggers for your anger.",
            "Express your feelings assertively without aggression.",
            "Engage in physical activity to release tension.",
            "Practice empathy by trying to understand other perspectives.",
            "Therapy can provide effective strategies for managing anger."
        ],
        "resources": [
            "Anger management techniques",
            "Communication skills"
        ]
    },
    "loneliness": {
        "possible_concerns": ["Social Isolation", "Feelings of Loneliness"],
        "advice": [
            "Reach out to old friends or family members.",
            "Join a club or group with shared interests (online or in person).",
            "Volunteer for a cause you care about.",
            "Practice small acts of kindness to connect with others.",
            "Remember that many people experience loneliness, and it's okay to seek connection."
        ],
        "resources": [
            "Building social connections",
            "Community resources"
        ]
    },
    "grief": {
        "possible_concerns": ["Bereavement", "Loss"],
        "advice": [
            "Allow yourself to grieve, there's no 'right' way or timeline for it.",
            "Lean on your support system of friends and family.",
            "Remember to take care of your physical health during this time.",
            "Find healthy ways to express your emotions.",
            "Consider joining a bereavement support group.",
            "A therapist specializing in grief can offer guidance and support."
        ],
        "resources": [
            "Grief counseling options",
            "Understanding the stages of grief"
        ]
    },
    "overthinking": {
        "possible_concerns": ["Rumination", "Excessive Worry"],
        "advice": [
            "Set aside a 'worry time' each day to focus on concerns, then let them go.",
            "Engage in activities that require focus and distract you (e.g., puzzles, hobbies).",
            "Practice mindfulness to bring your attention to the present moment.",
            "Challenge negative thought patterns.",
            "Journaling can help you process and release thoughts.",
            "A mental health professional can teach you specific techniques to manage overthinking."
        ],
        "resources": [
            "Cognitive restructuring techniques",
            "Mindfulness exercises for overthinking"
        ]
    },
    "burnout": {
        "possible_concerns": ["Exhaustion", "Reduced Performance", "Cynicism"],
        "advice": [
            "Prioritize self-care and rest.",
            "Evaluate your workload and responsibilities; look for areas to delegate or reduce.",
            "Set clear boundaries between work and personal life.",
            "Reconnect with activities that bring you joy and relaxation.",
            "Talk to your supervisor or HR if work is a major factor.",
            "Therapy can help you develop strategies for prevention and recovery from burnout."
        ],
        "resources": [
            "Burnout prevention strategies",
            "Work-life balance resources"
        ]
    },
    "self-harm": {
        "possible_concerns": ["Self-Injurious Behavior"],
        "advice": [
            "It sounds like you're going through a lot. Please reach out for immediate help. You don't have to face this alone.",
            "Distraction techniques can sometimes help in the moment: hold ice, snap a rubber band on your wrist, draw on your skin with a red marker, or listen to loud music.",
            "Identify your triggers and try to avoid them or develop coping strategies for them.",
            "A mental health professional can provide strategies and support. This is not something you have to deal with on your own."
        ],
        "resources": [
            "Crisis Hotlines (e.g., 988 in the US)",
            "Urgent Mental Health Care",
            "Therapy specializing in self-harm"
        ]
    },
    "panic attack": {
        "possible_concerns": ["Panic Disorder"],
        "advice": [
            "Focus on your breath. Breathe in slowly through your nose for 4 counts, hold for 7, and exhale slowly through your mouth for 8.",
            "Remind yourself that this feeling will pass. It's intense, but not dangerous.",
            "Engage in grounding techniques: look around and name 5 things you can see, 4 things you can feel, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
            "Splash cold water on your face or hold an ice pack.",
            "Move to a quiet, safe space.",
            "Learning coping mechanisms with a therapist can be very effective."
        ],
        "resources": [
            "Panic attack coping strategies",
            "Mindfulness for panic",
            "Therapy for panic disorder"
        ]
    }
}


# --- 2. Data Loading and Processing (RAG Setup) ---

# Check if the data directory exists and contains PDFs
if not os.path.exists(DATA_DIRECTORY) or not any(fname.endswith('.pdf') for fname in os.listdir(DATA_DIRECTORY)):
    st.error(f"No PDF documents found in '{DATA_DIRECTORY}'. Please add mental health PDFs to this folder.")
    st.stop()

# Load the PDF files from the path
loader = DirectoryLoader(DATA_DIRECTORY, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

if not documents:
    st.error(f"Failed to load any documents from '{DATA_DIRECTORY}'. Ensure PDFs are valid and readable.")
    st.stop()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # Increased chunk size for more context
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
# Ensure you have an internet connection for the first run to download the model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

# Create vectorstore
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Create LLM
# Make sure the Llama-2 model file is in your project directory
if not os.path.exists(LLAMA_MODEL_PATH):
    st.error(f"Llama-2 model '{LLAMA_MODEL_PATH}' not found. Please download it and place it in the project root.")
    st.stop()

llm = CTransformers(model=LLAMA_MODEL_PATH, model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.01, 'context_length': 2048}) # Increased tokens and context

# Initialize ConversationBufferMemory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Increased k for more retrieved documents
                                              memory=memory)

# --- 3. Streamlit Application Interface ---

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
    *   **[Your Local Emergency Services]:** e.g., 911 (for immediate danger)
    
    *Please replace the bracketed link with a relevant local resource for India or your region.*
    """
)

# --- 4. Conversation Logic ---

def conversation_chat(query):
    query_lower = query.lower()
    response = ""

    # HIGH PRIORITY: Crisis Detection
    crisis_keywords = [
        "suicide", "harm myself", "end my life", "kill myself", "hopeless",
        "can't go on", "want to die", "crisis", "self-harm"
    ]
    if any(keyword in query_lower for keyword in crisis_keywords):
        return (
            "I'm really sorry to hear you're feeling this way. It sounds incredibly difficult, "
            "and I want you to know you're not alone. My purpose is to provide support, "
            "but I'm not a substitute for immediate professional help during a crisis.\n\n"
            "**Please reach out to a crisis hotline or mental health professional immediately.** "
            "Here are some resources that can provide immediate support:\n"
            "📞 **National Suicide Prevention Lifeline (US): 988**\n"
            "📱 **Crisis Text Line (US): Text HOME to 741741**\n"
            "Please connect with someone who can provide immediate support."
        )

    # Mental Health Symptom Mapping (Specific, curated advice)
    for symptom, details in mental_health_mapping.items():
        if symptom in query_lower:
            concerns = ", ".join(details["possible_concerns"])
            advice_list = details["advice"]
            
            response_parts = [
                f"It sounds like you might be experiencing {concerns}. Here's some general advice that might help:"
            ]
            for i, advice_point in enumerate(advice_list):
                response_parts.append(f"• {advice_point}")
            
            response_parts.append(
                "\nRemember, these are general tips, and talking to a professional can provide tailored support."
            )
            response = "\n".join(response_parts)
            break # Only respond to the first matching mental health symptom for brevity

    # Fallback to LLM for broader knowledge retrieval if no specific mapping match
    if not response:
        try:
            # We add a "persona" to the question for the LLM
            llm_question = f"As a supportive mental wellness assistant, provide information and coping strategies for: {query}"
            result = chain({"question": llm_question, "chat_history": st.session_state['history']})
            response = result["answer"]
        except Exception as e:
            st.error(f"An error occurred with the LLM. Please try again or recheck your model setup. Error: {e}")
            response = "I'm having trouble processing that right now. Could you rephrase or ask something else?"

    # Update chat history
    st.session_state['history'].append((query, response))
    return response

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I'm MindMentor, your mental wellness bot. How can I support you today? 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey there! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Tell me what's on your mind:", placeholder="E.g., I'm feeling stressed about work.", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                # You can customize the bot's avatar style too!
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()