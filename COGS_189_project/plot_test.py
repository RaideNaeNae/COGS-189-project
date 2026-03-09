import numpy as np
import matplotlib.pyplot as plt

# UPDATE THIS PATH to the newest folder in your directory!
file_path = r"data\cyton_participant-www\ses-001\2026-03-08_22h43.34.367\eeg_raw.npy" 

# Load the data
eeg_data = np.load(file_path, allow_pickle=True)

# Print the shape to verify data exists
print(f"Data shape loaded: {eeg_data.shape}")

# Check if the array is empty
if eeg_data.shape[1] == 0:
    print("ERROR: The file is empty! You pressed Escape too quickly.")
else:
    # Plot Channel 1
    plt.plot(eeg_data[0, :])
    plt.title("Synthetic EEG Data - Channel 1")
    plt.xlabel("Samples (250Hz)")
    plt.ylabel("Amplitude")
    plt.show()