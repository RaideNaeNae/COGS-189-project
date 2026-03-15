import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Select particpant specific data
participant_name = "angelo" 

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

# 1 = Classical, 2 = EDM
trigger_order = np.array([2, 1, 1, 2, 2, 1, 2, 1, 1, 2]) 

# Create MNE Epochs
epochs = mne.EpochsArray(epochs_data, info)
epochs.filter(l_freq=1.0, h_freq=59.0)

# Calculate PSD with 0.5s windows and 50% overlap
psds = epochs.compute_psd(
    method='welch', 
    n_fft=256,        # Keep n_fft at 256 for better frequency resolution
    n_per_seg=125,    # 0.5 seconds
    n_overlap=62,     # 0.25 seconds
    fmin=8, 
    fmax=30
)
psd_data = psds.get_data() # (Trials, Channels, Freqs)
freqs = psds.freqs

# Define Bands
bands = {'Alpha': (8, 12), 'Beta': (13, 30)}

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
pivot_df = pivot_df.reindex(['Alpha', 'Beta'])

ax = pivot_df.plot(kind='bar', figsize=(8, 6), color=['#1f77b4', '#ff7f0e'])
plt.title("PSD: Classical vs. EDM (Angelo)", fontsize=14)
plt.ylabel("Mean Power")
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

from scipy import stats

# 1. Get the raw power for each of the 10 trials for a specific band (e.g., Beta)
# We need the mean power per trial, not just the overall average
trial_means_classical = []
trial_means_edm = []

# Assuming 'psd_data' is (10 trials, 8 channels, freqs)
for i in range(10):
    genre_id = trigger_order[i]
    # Find frequencies in Beta range (13-30Hz)
    idx = np.logical_and(freqs >= 13, freqs <= 30)
    # Average across channels and frequencies for this specific trial
    mean_power = psd_data[i, :, idx].mean()
    
    if genre_id == 1:
        trial_means_classical.append(mean_power)
    else:
        trial_means_edm.append(mean_power)

# 2. Run the Paired T-Test
t_stat, p_val = stats.ttest_rel(trial_means_classical, trial_means_edm)

print(f"\n--- Within-Subject T-Test (Beta Band) ---")
print(f"Classical Trial Means: {np.round(trial_means_classical, 13)}")
print(f"EDM Trial Means:       {np.round(trial_means_edm, 13)}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_val:.4f}")

if p_val < 0.05:
    print("RESULT: Statistically Significant! The genre significantly changed Beta power.")
else:
    print("RESULT: Not Significant. The difference could be due to noise.")