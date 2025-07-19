import torch
import torch.nn as nn
import os
import streamlit as st
from models import Voc, EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, device

def loadCheckpoint(checkpoint_path, corpus_name, hidden_size, encoder_n_layers, decoder_n_layers, dropout, attn_model):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc_dict = checkpoint['voc_dict']
    
    voc = Voc(corpus_name)
    voc.__dict__ = voc_dict
    
    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    
    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    encoder.eval()
    decoder.eval()
    
    return voc, encoder, decoder

@st.cache_resource
def load_model():
    save_dir = os.path.join("data", "save")
    model_name = 'chatbot_model'
    corpus_name = "movie-corpus"
    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    
    directory = os.path.join(save_dir, model_name, corpus_name, 
                            f'{encoder_n_layers}-{decoder_n_layers}_{hidden_size}')
    
    if not os.path.exists(directory):
        st.error("No checkpoint directory found. Please train the model first.")
        return None, None, None, None
        
    checkpoint_files = [f for f in os.listdir(directory) if f.endswith('_checkpoint.tar')]
    if not checkpoint_files:
        st.error("No checkpoint files found. Please train the model first.")
        return None, None, None, None
    
    iterations = [int(f.split('_')[0]) for f in checkpoint_files]
    max_iteration = max(iterations)
    checkpoint_path = os.path.join(directory, f"{max_iteration}_checkpoint.tar")
    
    try:
        voc, encoder, decoder = loadCheckpoint(checkpoint_path, corpus_name, hidden_size, 
                                             encoder_n_layers, decoder_n_layers, dropout, attn_model)
        searcher = GreedySearchDecoder(encoder, decoder)
        return voc, encoder, decoder, searcher
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None