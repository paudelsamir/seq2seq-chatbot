import streamlit as st
from model_loader import load_model
from utils import get_response

def main():
    st.set_page_config(
        page_title="Chatbot",
        page_icon="📀",
        layout="wide"
    )
    
    # Custom CSS for modern styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }

    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">AI Dialogue Chatbot</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">A neural network trained on the <a href="https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html" target="_blank" style="color:#e63946;text-decoration:underline;">Cornell Movie-Dialogs Corpus</a></p>',
        unsafe_allow_html=True
    )
    
    # Load model
    with st.spinner("Loading model..."):
        voc, encoder, decoder, searcher = load_model()
    
    if voc is None:
        st.error("Failed to load the model")
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm an AI trained on dialogue data. Feel free to start a conversation with me."
        })
    
    # Chat container
    with st.container():
        # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Generating response..."):
            response = get_response(prompt, voc, encoder, decoder, searcher)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    


    # Sidebar
    with st.sidebar:
            # Footer (left-aligned, red link)
        st.markdown(
            """
            <div style="text-align:left; color:#aaa; font-size:0.85rem; margin-top:2.5rem;">
            Made by <a href="https://x.com/samireey" target="_blank" style="color:#e63946;text-decoration:underline;">@samireey</a>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        # st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### Quick Start")
        st.markdown("""
        <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem;">
            <b>Try these conversation starters:</b>
            <ul style="margin-top:0.4rem; margin-bottom:0.4rem;">
            <li>"Hello there!"</li>
            <li>"How are you?"</li>
            <li>"Tell me something interesting"</li>
            <li>"What's your opinion on..."</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### About")
        st.markdown("""
        <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem;">
            <div><b>Architecture:</b> Seq2Seq + Luong Attention</div>
            <div><b>Data:</b> Cornell Movie-Dialogs Corpus</div>
            <div><b>Model:</b> Encoder-Decoder (GRU)</div>
            <div><b>Vocab:</b> ~7K words</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        


        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = []
            st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm an AI trained on dialogue data. Feel free to start a conversation with me."
            })
            st.rerun()
        with st.expander("Model Limitations & Improvements", expanded=False):
            st.markdown("""
            <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem;">
            we built a basic seq2seq chatbot trained on movie dialogues. while it can generate responses, it's limited by a short input length (10 words), a small vocabulary (~7k words), and greedy decoding—leading to repetitive, dull outputs. it lacks memory, so every reply is context-free. it's also prone to crashing on unknown words and can't handle emotion, sentiment, or real multi-turn conversations. training data is narrow (only film scripts) and shallow (just 4k iterations), which means it doesn’t generalize well to natural, real-world chats.<br><br>
            to improve it, we can start by increasing max input length, using temperature sampling, and gracefully handling unknown words. adding beam search, short-term memory, and pre-trained embeddings (like GloVe) would boost quality significantly. switching to a transformer model unlocks multi-turn conversations, emotion recognition, and more natural replies. pairing that with real-world data, filtering toxic outputs, and integrating factual knowledge or RLHF would make the bot smarter, safer, and more human-like.
            </div>
            """, unsafe_allow_html=True)

        # Footer (centered, subtle)

if __name__ == "__main__":
    main()