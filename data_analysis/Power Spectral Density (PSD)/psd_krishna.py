import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Select particpant specific data
participant_name = "krishna" 

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"
matching_folders = [f for f in data_dir.glob(f"*{participant_name}*") if f.is_dir()]
target_folder = matching_folders[0]
target_file = list(target_folder.rglob("eeg_trials.npy"))
file_path = list(target_folder.rglob("eeg_trials.npy"))[0]

epochs_data = np.load(file_path, allow_pickle=True) * 1e-6 

sfreq = 250
ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg']*8)


sfreq = 250
ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg']*8)

# 1 = Classical, 2 = EDM
trigger_order = np.array([2, 1, 2, 1, 2, 1, 2, 1, 2, 1]) 

# Create MNE Epochs
epochs = mne.EpochsArray(epochs_data, info)
epochs.filter(l_freq=1.0, h_freq=40.0)

# Calculate PSD
psds = epochs.compute_psd(method='welch', fmin=1, fmax=40)
psd_data = psds.get_data() # (Trials, Channels, Freqs)
freqs = psds.freqs

# Define Bands
bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (13, 30)}

# Group by Genre and Calculate Means
results = []
for genre_name, genre_id in [('Classical', 1), ('EDM', 2)]:
    # Pull only the trials for this genre
    genre_mask = (trigger_order == genre_id)
    genre_psds = psd_data[genre_mask] # Shape: (5 trials, 8 channels, freqs)
    
    for band_name, (fmin, fmax) in bands.items():
        # Find frequencies in range
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        # Average across those frequencies, then across channels, then across trials
        mean_power = genre_psds[:, :, idx].mean()
        results.append({'Genre': genre_name, 'Band': band_name, 'Power': mean_power})

# Visualization
df_results = pd.DataFrame(results)
pivot_df = df_results.pivot(index='Band', columns='Genre', values='Power')

# Reorder bands for the x-axis
pivot_df = pivot_df.reindex(['Delta', 'Theta', 'Alpha', 'Beta'])

ax = pivot_df.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
plt.title("Brainwave Power Comparison: Classical vs. EDM (Krishna)", fontsize=14)
plt.ylabel("Mean Power (uV^2/Hz)")
plt.xlabel("Frequency Band")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Music Genre")

plt.show()

print("\nSummary of Differences")
for band in ['Alpha', 'Beta']:
    c_pow = pivot_df.loc[band, 'Classical']
    e_pow = pivot_df.loc[band, 'EDM']
    diff = ((e_pow - c_pow) / c_pow) * 100
    direction = "increase" if diff > 0 else "decrease"
    print(f"{band} power showed a {abs(diff):.1f}% {direction} during EDM compared to Classical.")