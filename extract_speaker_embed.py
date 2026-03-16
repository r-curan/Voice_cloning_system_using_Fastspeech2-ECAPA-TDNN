import os
import torch
from torch.nn import functional as F
import yaml
from tqdm import tqdm

from audio.tools import load_audio_mono_16k, trim_silence
from model.pre_trained.ecapa_tdnn_loader import get_ECAPA_TDNN_MODEL, speaker_embedding_extractor

def extract_embeddings(config):
    dataset_dir = config["path"]["corpus_path"]
    model_name = "ecapa"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speaker_model = get_ECAPA_TDNN_MODEL(device=device)

    for speaker in tqdm(os.listdir(dataset_dir)):
        speaker_path = os.path.join(dataset_dir, speaker)

        if not os.path.isdir(speaker_path):
            continue

        embs = []

        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)

            if not os.path.isdir(chapter_path) or "_embedding" in chapter:
                continue 

            for file_name in os.listdir(chapter_path):
                if file_name[-4:] != ".wav":
                    continue 

                wav_path = os.path.join(chapter_path, file_name)

                try:
                    waveform = load_audio_mono_16k(wav_path)
                    waveform = trim_silence(waveform)

                    if waveform is None:
                        continue
                    emb = speaker_embedding_extractor(speaker_model, waveform)

                    # save per utterance
                    base_name = file_name[:-4]
                    utt_path = os.path.join(chapter_path, f"{base_name}.{model_name}_embedding")
                    torch.save(emb, utt_path)

                    embs.append(emb)

                except Exception as e:
                    print("Error: ", wav_path)
                    print(e)

        if embs:
            avg_emb = torch.mean(torch.stack(embs), dim=0)
            # avg_emb = F.normalize(avg_emb, p=2, dim=1)
            avg_path = os.path.join(speaker_path, f"{speaker}.{model_name}_averaged_embedding")
            torch.save(avg_emb, avg_path)

        else:
            print(f"Warning! Speaker ID: {speaker} have all very short utterance so embedding not considered on averaged.")

if __name__ == "__main__":
    config_path = "config/preprocess.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    extract_embeddings(config)
