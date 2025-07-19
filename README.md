A sequence-to-sequence neural network chatbot trained on the Cornell Movie-Dialogs Corpus. This project implements an encoder-decoder architecture with attention mechanism to generate conversational responses.


## Live Demo

Try the chatbot live: **[seq2seq-chatbot-samir.streamlit.app](https://seq2seq-chatbot-samir.streamlit.app)**

[<video controls src="demo.mp4" title="Title"></video>](https://github.com/user-attachments/assets/34fbf8b2-b575-407a-9b67-57ec68c59514)

##  Model Architecture

- **Encoder**: Bidirectional GRU (2 layers, 500 hidden units)
- **Decoder**: Unidirectional GRU with attention (2 layers, 500 hidden units)
- **Attention**: Luong dot-product attention mechanism
- **Vocabulary**: ~7,000 words from movie dialogues
- **Training**: 4,000 iterations with teacher forcing

##  Model Download
The trained model (~300MB) is available on:
#### Option 1:  [Download from Google Drive](https://drive.google.com/file/d/1DFdRoXweBM0qsrg9Uwg58_pICwOUlWjt/view?usp=drive_link)
#### Option 2:  [Download from Hugging Face](https://huggingface.co/paudelsamir/my-checkpoints/blob/main/4000_checkpoint.tar)

*Note: The app automatically downloads from Google Drive when deployed on Streamlit Cloud.*

## Installation - Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/paudelsamir/seq2seq-chatbot.git
   cd seq2seq-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model**
   - Download from [Google Drive](https://drive.google.com/file/d/1DFdRoXweBM0qsrg9Uwg58_pICwOUlWjt/view?usp=drive_link)
   - Place `4000_checkpoint.tar` in: `data/save/chatbot_model/movie-corpus/2-2_500/`

4. **Run the app**
   ```bash
   streamlit run app.py
   ```


## 📁 Project Structure

```
seq2seq-chatbot/
├── app.py                 # Streamlit web application
├── model_loader.py        # Model loading and downloading utilities
├── models.py             # Neural network architecture definitions
├── utils.py              # Text processing and response generation
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── data/                # Model storage directory
    └── save/
        └── chatbot_model/
            └── movie-corpus/
                └── 2-2_500/
                    └── 4000_checkpoint.tar
```

