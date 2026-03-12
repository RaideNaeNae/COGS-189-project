import numpy as np
import mne
import matplotlib.pyplot as plt

# 1. Load your data
epochs_data = np.load(R"data\cyton_participant-angelo_real_test\ses-001\2026-03-09_15h28.35.822\eeg_trials.npy", allow_pickle=True) * 1e-6 

# 2. DEFINE YOUR CHANNELS 
# If you get a "channels missing" error, double check these 8 names match your cap
ch_names = ['Fz', 'Cz', 'Pz', 'T3', 'T4', 'T5', 'T6', 'O1'] 
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=['eeg']*8)

# 3. Create the Epochs and set the Montage
trigger_order = [2, 1, 1, 2, 2, 1, 2, 1, 1, 2] 
event_dict = {'Classical': 1, 'EDM': 2}
events = np.zeros((10, 3), dtype=int)
events[:, 0] = np.arange(10)
events[:, 2] = trigger_order

epochs = mne.EpochsArray(epochs_data, info, events=events, event_id=event_dict)
montage = mne.channels.make_standard_montage('standard_1020')
epochs.set_montage(montage)

# 4. Filter and compute Alpha (8-12 Hz)
epochs.filter(l_freq=1.0, h_freq=40.0)
psds = epochs.compute_psd(fmin=8, fmax=12)
psd_data = psds.get_data() 

# 5. Average the Alpha power per genre
classical_alpha = psd_data[np.array(trigger_order) == 1].mean(axis=(0, 2))
edm_alpha = psd_data[np.array(trigger_order) == 2].mean(axis=(0, 2))

# ---------------------------------------------------------
# 6. PLOT: Two Labeled Heatmaps Side-by-Side
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plot Classical
# Removed 'show_names' to fix the TypeError. 
# 'names' handles the labels, and 'sensors=True' ensures the dots show up.
mne.viz.plot_topomap(classical_alpha, epochs.info, axes=ax1, 
                     show=False, contours=4, names=ch_names, sensors=True)
ax1.set_title("CLASSICAL: Alpha Power (8-12 Hz)", fontsize=14, pad=20)

# Plot EDM
im2, _ = mne.viz.plot_topomap(edm_alpha, epochs.info, axes=ax2, 
                             show=False, contours=4, names=ch_names, sensors=True)
ax2.set_title("EDM: Alpha Power (8-12 Hz)", fontsize=14, pad=20)

# Add a shared colorbar
# im2[0] is used to grab the map object for the colorbar
cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
fig.colorbar(ax2.images[0], cax=cax, label='Power ($\mu V^2/Hz$)')

plt.suptitle("Spatial Comparison of Music Genre Effects", fontsize=18, fontweight='bold')
plt.show()