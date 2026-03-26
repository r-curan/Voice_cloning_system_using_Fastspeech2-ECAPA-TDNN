import torch
import torchaudio
from model.pre_trained import get_ECAPA_TDNN_MODEL, speaker_embedding_extractor
import os

def generate_embedding(audio_path, output_filename):
    """Generate PROPERLY NORMALIZED ECAPA-TDNN embedding"""
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_ECAPA_TDNN_MODEL(device=device)
    
    # Load and preprocess audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    print(f"Audio loaded: {waveform.shape}, sr: {sr}")
    
    # Extract embedding using YOUR function
    embedding = speaker_embedding_extractor(model, waveform)
    
    # **CRITICAL**: L2 NORMALIZE (like your original get_speaker_emb)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    print(f"AFTER NORMALIZATION - Min: {embedding.min():.4f}, Max: {embedding.max():.4f}, Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
    
    # Save
    torch.save(embedding.cpu(), output_filename)
    print(f"Saved NORMALIZED embedding: {output_filename} (shape: {embedding.shape})")

# if __name__ == "__main__":
#     AUDIO_PATH = "embeddings/LibriTTS/7611_processed.wav"  # Path to your input audio file
#     OUTPUT_NAME = "embeddings/LibriTTS/7611.pt"
    
#     generate_embedding(AUDIO_PATH, OUTPUT_NAME)
