import torch
import unicodedata
import re
from models import PAD_token, SOS_token, EOS_token, MAX_LENGTH, device

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word, 0) for word in sentence.split(' ')] + [EOS_token]

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    
    try:
        decoded_words = decoded_words[:decoded_words.index('EOS')]
    except ValueError:
        pass
    
    return decoded_words

def get_response(input_text, voc, encoder, decoder, searcher):
    try:
        normalized_input = normalizeString(input_text)
        if not normalized_input.strip():
            return "I don't understand."
        
        output_words = evaluate(encoder, decoder, searcher, voc, normalized_input)
        output_words = [word for word in output_words if word not in ['EOS', 'PAD']]
        return ' '.join(output_words) if output_words else "I don't understand."
    except KeyError:
        return "Sorry, I don't understand some words in your message."
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"