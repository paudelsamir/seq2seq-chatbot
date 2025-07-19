import torch
import torch.nn as nn
import os
import streamlit as st
import requests
from models import Voc, EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder, device

def download_from_google_drive(file_id, local_path):
    """Download file from Google Drive using file ID"""
    # Use the direct download URL for Google Drive
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download with session to handle redirects
    with st.spinner("Downloading model from Google Drive..."):
        session = requests.Session()
        
        # First request to get the download page
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Check if we need to handle the virus scan warning
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                # Get the confirmation URL
                confirm_url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                response = session.get(confirm_url, stream=True)
                break
        
        # Verify we got a valid file (not HTML)
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            # Try alternative download method
            response = session.get(f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t", stream=True)
        
        # Save the file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify the file is a valid tar file
        try:
            import tarfile
            with tarfile.open(local_path, 'r') as tar:
                tar.getnames()  # Just check if it's a valid tar file
            print(f"Successfully downloaded and verified model: {local_path}")
        except Exception as e:
            os.remove(local_path)  # Remove invalid file
            raise Exception(f"Downloaded file is not a valid checkpoint: {e}")
        
        print(f"Downloaded model to: {local_path}")

def download_checkpoint_if_missing(google_drive_file_id, directory):
    filename = "4000_checkpoint.tar"
    local_path = os.path.join(directory, filename)

    if not os.path.exists(local_path):
        try:
            print(f"Downloading checkpoint from Google Drive...")
            download_from_google_drive(google_drive_file_id, local_path)
            print(f"Successfully downloaded checkpoint to: {local_path}")
        except Exception as e:
            st.error(f"❌ Failed to download model from Google Drive: {e}")
            st.error("Please check if the Google Drive file is publicly accessible")
            raise e
    else:
        print(f"Checkpoint already exists at: {local_path}")

def loadCheckpoint(checkpoint_path, corpus_name, hidden_size, encoder_n_layers, decoder_n_layers, dropout, attn_model):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
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

    # Google Drive file ID for your checkpoint
    google_drive_file_id = "1DFdRoXweBM0qsrg9Uwg58_pICwOUlWjt"
    download_checkpoint_if_missing(google_drive_file_id, directory)

    if not os.path.exists(directory):
        st.error("No checkpoint directory found.")
        return None, None, None, None
        
    checkpoint_files = [f for f in os.listdir(directory) if f.endswith('_checkpoint.tar')]
    if not checkpoint_files:
        st.error("No checkpoint files found.")
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