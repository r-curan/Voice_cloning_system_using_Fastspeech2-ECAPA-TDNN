from preprocessor.preprocessor import get_mel_from_wav_numpy,compute_energy_numpy,compute_pitch_numpy,load_wav_numpy
import numpy as np
from embeddinggenerator import generate_embedding
import torch


# path
wav=load_wav_numpy("similarity/original/8887_281471_000000_000000.wav",22050)
original_embedding=torch.load("similarity/original/8887.pt")
synth=load_wav_numpy("similarity/synthesized/8887_281471_000000_000000.wav",22050)
synth_embedding=generate_embedding("similarity/synthesized/8887_281471_000000_000000.wav","similarity/synthesized/8887_synth.pt")
synth_embedding=torch.load("similarity/synthesized/8887_synth.pt")



mel1=get_mel_from_wav_numpy(wav)
energy1=compute_energy_numpy(wav,hop_length=256)    
pitch1=compute_pitch_numpy(wav,sr=22050) 

mel2=get_mel_from_wav_numpy(synth)
energy2=compute_energy_numpy(synth,hop_length=256)    
pitch2=compute_pitch_numpy(synth,sr=22050)
    

def cosine_similarity_pct(e1: torch.Tensor, e2: torch.Tensor) -> float:
    e1 = torch.nn.functional.normalize(e1.view(1, -1).float(), p=2, dim=1)
    e2 = torch.nn.functional.normalize(e2.view(1, -1).float(), p=2, dim=1)
    cos = float((e1 * e2).sum())
    return round((cos + 1) / 2 * 100, 2)

def compute_overall_similarity(mel1, mel2, energy1, energy2, pitch1, pitch2):
    # Compute cosine similarity for mel spectrograms
    mel_similarity = np.dot(mel1.flatten(), mel2.flatten()) / (
        np.linalg.norm(mel1.flatten()) * np.linalg.norm(mel2.flatten())
    )
    
    # Compute mean absolute error for energy
    energy_similarity = 1 - np.mean(np.abs(energy1 - energy2) / (np.max(energy1) + 1e-8))
    
    # Compute mean absolute error for pitch
    pitch_similarity = 1 - np.mean(np.abs(pitch1 - pitch2) / (np.max(pitch1) + 1e-8))
    
    # # Combine similarities (you can adjust weights as needed)
    # overall_similarity = (mel_similarity + energy_similarity + pitch_similarity) / 3
    
    return  mel_similarity, energy_similarity, pitch_similarity





a,b,c=compute_overall_similarity(mel1, mel2, energy1, energy2, pitch1, pitch2)
embedding_similarity=cosine_similarity_pct(original_embedding, synth_embedding)


print("Mel Similarity:", a)
print("Energy Similarity:", b)
print("Pitch Similarity:", c)
print("Embedding Similarity:", embedding_similarity)
  