import numpy as np
import mne 
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

#setup
participants = ["angelo", "krishna", "pedro"]
script_dir = Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"

sfreq = 250
ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg']*8)
trigger_order = np.array([2, 1, 2, 1, 2, 1, 2, 1, 2, 1]) 

bands = {'Alpha': (8, 12), 'Beta': (13, 30)}
all_results = {}

for p_name in participants:
    try:
        match = [f for f in data_dir.glob(f"*{p_name}*") if f.is_dir()][0]
        file_path = list(match.rglob("eeg_trials.npy"))[0]
        
        data = np.load(file_path, allow_pickle=True) * 1e-6 
        epochs = mne.EpochsArray(data, info, verbose=False)
        epochs.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        psds = epochs.compute_psd(
            method='welch', n_fft=256, n_per_seg=125, n_overlap=62, 
            fmin=8, fmax=30, verbose=False 
        )
        psd_data = psds.get_data()
        freqs = psds.freqs
        
        p_res = []
        for genre_name, genre_id in [('Classical', 1), ('EDM', 2)]:
            mask = (trigger_order == genre_id)
            for b_name, (fmin, fmax) in bands.items():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                mean_pow = psd_data[mask][:, :, idx].mean()
                p_res.append({'Band': b_name, 'Genre': genre_name, 'Power': mean_pow})
        
        all_results[p_name] = pd.DataFrame(p_res).pivot(index='Band', columns='Genre', values='Power').reindex(['Alpha', 'Beta'])
        print(f"✅ Processed {p_name}")
    except Exception as e:
        print(f"❌ Failed to process {p_name}: {e}")

#scaling (based on Alpha/Beta only)
global_max = max([df.values.max() for df in all_results.values()]) * 1.1

#plotting
fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

for i, p_name in enumerate(participants):
    if p_name in all_results:
        all_results[p_name].plot(kind='bar', ax=axes[i], color=['#1f77b4', '#ff7f0e'], legend=(i==2))
        axes[i].set_title(f"Participant: {p_name.upper()}")
        axes[i].set_xlabel("Frequency Band")
        axes[i].set_xticklabels(['Alpha', 'Beta'], rotation=0)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

axes[0].set_ylabel("Mean Power")
plt.ylim(0, global_max)
plt.tight_layout()

plt.show()