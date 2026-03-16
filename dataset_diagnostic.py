import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity

PREPROCESSED = "preprocessed_data/LibriTTS"
RAW_AUDIO = "raw_data/LibriTTS"
ECAPA_PATH = "datasets/LibriTTS"

PITCH_DIR = os.path.join(PREPROCESSED,"pitch")
ENERGY_DIR = os.path.join(PREPROCESSED,"energy")
DURATION_DIR = os.path.join(PREPROCESSED,"duration")
MEL_DIR = os.path.join(PREPROCESSED,"mel")

os.makedirs("plots",exist_ok=True)
os.makedirs("reports",exist_ok=True)

print("Scanning dataset...")

speaker_pitch = defaultdict(list)
speaker_energy = defaultdict(list)
speaker_duration = defaultdict(list)
speaker_mel_frames = defaultdict(list)

all_pitch=[]
all_energy=[]
all_duration=[]
mel_lengths=[]

bad_samples=[]

files=os.listdir(PITCH_DIR)

for f in tqdm(files):

    spk=f.split("-")[0]
    base=f.split("-")[-1].replace(".npy","")

    pitch=np.load(os.path.join(PITCH_DIR,f))
    energy=np.load(os.path.join(ENERGY_DIR,f.replace("pitch","energy")))
    duration=np.load(os.path.join(DURATION_DIR,f.replace("pitch","duration")))
    mel=np.load(os.path.join(MEL_DIR,f.replace("pitch","mel")))

    speaker_pitch[spk].extend(pitch)
    speaker_energy[spk].extend(energy)
    speaker_duration[spk].extend(duration)
    speaker_mel_frames[spk].append(mel.shape[0])

    all_pitch.extend(pitch)
    all_energy.extend(energy)
    all_duration.extend(duration)
    mel_lengths.append(mel.shape[0])

    issues=[]

    if np.all(pitch==0):
        issues.append("pitch_zero")

    if np.std(pitch)>3:
        issues.append("pitch_variance_explosion")

    if np.max(np.abs(pitch))>5:
        issues.append("pitch_outlier")

    if np.mean(energy)<-3:
        issues.append("low_energy")

    if np.std(energy)>3:
        issues.append("energy_variance_explosion")

    if np.max(duration)>40:
        issues.append("duration_outlier")

    if mel.shape[0]<50:
        issues.append("very_short_utterance")

    if abs(np.sum(duration)-mel.shape[0])>5:
        issues.append("alignment_mismatch")

    if len(issues)>0:

        bad_samples.append({
            "file":f,
            "speaker":spk,
            "audio":os.path.join(RAW_AUDIO,spk,base+".wav"),
            "issues":issues
        })

all_pitch=np.array(all_pitch)
all_energy=np.array(all_energy)
all_duration=np.array(all_duration)
mel_lengths=np.array(mel_lengths)

print("\nDataset stats")
print("Utterances:",len(files))
print("Speakers:",len(speaker_pitch))

print("Pitch mean",np.mean(all_pitch))
print("Pitch std",np.std(all_pitch))

print("Energy mean",np.mean(all_energy))
print("Energy std",np.std(all_energy))

print("Duration mean",np.mean(all_duration))
print("Duration std",np.std(all_duration))

print("Mel mean",np.mean(mel_lengths))

# pitch distribution
plt.figure()
plt.hist(all_pitch,bins=120)
plt.title("Pitch Distribution")
plt.xlabel("Normalized Pitch")
plt.ylabel("Samples")
plt.savefig("plots/pitch_hist.png")
plt.close()

# energy distribution
plt.figure()
plt.hist(all_energy,bins=120)
plt.title("Energy Distribution")
plt.xlabel("Normalized Energy")
plt.ylabel("Samples")
plt.savefig("plots/energy_hist.png")
plt.close()

# duration distribution
plt.figure()
plt.hist(all_duration,bins=120)
plt.title("Phoneme Duration Distribution")
plt.xlabel("Duration (mel frames)")
plt.ylabel("Phonemes")
plt.savefig("plots/duration_hist.png")
plt.close()

# mel lengths
plt.figure()
plt.hist(mel_lengths,bins=120)
plt.title("Mel Length Distribution")
plt.xlabel("Mel frames per utterance")
plt.ylabel("Utterances")
plt.savefig("plots/mel_length_hist.png")
plt.close()

# speaker sample counts
speaker_counts={s:len(speaker_mel_frames[s]) for s in speaker_mel_frames}

sorted_items=sorted(speaker_counts.items(),key=lambda x:x[1])

spk=[x[0] for x in sorted_items]
cnt=[x[1] for x in sorted_items]

plt.figure(figsize=(10,5))
plt.bar(spk,cnt)
plt.xticks(rotation=90)
plt.title("Utterances per Speaker")
plt.xlabel("Speaker")
plt.ylabel("Number of Utterances")
plt.savefig("plots/speaker_sample_count.png")
plt.close()

# speaker pitch
pitch_stats=[(s,np.mean(speaker_pitch[s])) for s in speaker_pitch]
pitch_stats=sorted(pitch_stats,key=lambda x:x[1])

spk=[x[0] for x in pitch_stats]
vals=[x[1] for x in pitch_stats]

plt.figure(figsize=(10,5))
plt.bar(spk,vals)
plt.xticks(rotation=90)
plt.title("Speaker Mean Pitch")
plt.xlabel("Speaker")
plt.ylabel("Mean Normalized Pitch")
plt.savefig("plots/speaker_pitch_range.png")
plt.close()

# speaker energy
energy_stats=[(s,np.mean(speaker_energy[s])) for s in speaker_energy]
energy_stats=sorted(energy_stats,key=lambda x:x[1])

spk=[x[0] for x in energy_stats]
vals=[x[1] for x in energy_stats]

plt.figure(figsize=(10,5))
plt.bar(spk,vals)
plt.xticks(rotation=90)
plt.title("Speaker Mean Energy")
plt.xlabel("Speaker")
plt.ylabel("Mean Normalized Energy")
plt.savefig("plots/speaker_energy_range.png")
plt.close()

# pitch outliers
pitch_mean=np.mean(all_pitch)
pitch_std=np.std(all_pitch)

outliers=np.abs(all_pitch-pitch_mean)>4*pitch_std

plt.figure()
plt.hist(all_pitch[outliers],bins=100)
plt.title("Extreme Pitch Outliers")
plt.xlabel("Pitch")
plt.ylabel("Samples")
plt.savefig("plots/outlier_pitch.png")
plt.close()

# ecapa embeddings
print("\nChecking ECAPA embeddings")

embeddings=[]
names=[]
norms=[]

for spk in speaker_pitch:

    emb_path=os.path.join(ECAPA_PATH,spk,f"{spk}.ecapa_averaged_embedding")

    if os.path.exists(emb_path):

        emb=torch.load(emb_path,map_location="cpu")

        if isinstance(emb,torch.Tensor):
            emb=emb.detach().cpu().numpy()

        emb=emb.squeeze()

        embeddings.append(emb)
        names.append(spk)
        norms.append(np.linalg.norm(emb))

norms=np.array(norms)

print("Embedding norm mean",np.mean(norms))
print("Embedding norm std",np.std(norms))

plt.figure()
plt.hist(norms,bins=50)
plt.title("ECAPA Embedding Norm Distribution")
plt.xlabel("L2 norm")
plt.ylabel("Speakers")
plt.savefig("plots/ecapa_norms.png")
plt.close()

embeddings=np.vstack(embeddings)

sim=cosine_similarity(embeddings)

plt.figure(figsize=(8,8))
plt.imshow(sim)
plt.colorbar()
plt.title("Speaker Embedding Similarity")
plt.xlabel("Speaker index")
plt.ylabel("Speaker index")
plt.savefig("plots/ecapa_similarity.png")
plt.close()

# detect duplicate speakers
duplicates=[]

for i in range(len(names)):
    for j in range(i+1,len(names)):
        if sim[i,j]>0.95:
            duplicates.append((names[i],names[j],sim[i,j]))

print("\nPossible duplicate speakers:",len(duplicates))

# save report
with open("reports/bad_samples.json","w") as f:
    json.dump(bad_samples,f,indent=2)

print("\nBad samples:",len(bad_samples))
print("Report saved to reports/bad_samples.json")

print("\nDiagnostics complete")
print("Plots -> plots/")
print("Reports -> reports/")