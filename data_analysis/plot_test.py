import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Select particpant specific data
participant_name = "krishna" 

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir / "data"
matching_folders = [f for f in data_dir.glob(f"*{participant_name}*") if f.is_dir()]
target_folder = matching_folders[0]
target_file = list(target_folder.rglob("eeg_raw.npy"))
file_path = list(target_folder.rglob("eeg_raw.npy"))[0]

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