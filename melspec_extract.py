import librosa
import numpy as np
import os
import pandas as pd

# Fixed duration in seconds
FIXED_DURATION = 4  # 4 seconds
SPEC_HEIGHT = 50  # Number of mel bands
SPEC_WIDTH = 50   # Number of time frames

# Function to load, pad/crop, and extract Mel spectrogram matrix
def extract_melspectrogram(file_path, fixed_duration=FIXED_DURATION):
    y, sr = librosa.load(file_path, sr=None)  # Load at native sampling rate
    target_length = sr * fixed_duration  # Desired number of samples

    # Crop or pad the audio to exactly 4 seconds
    if len(y) > target_length:
        y = y[:target_length]  # Truncate if too long
    else:
        y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')  # Pad with zeros

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=SPEC_HEIGHT)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels
    
    # Resize the spectrogram to 50x50
    mel_spec_resized = librosa.util.fix_length(mel_spec_db, size=SPEC_WIDTH, axis=1)  # Time-axis resizing

    return mel_spec_resized

# Path to the main audio directory
audio_dir = "E:\\BIJI\\DATASET"
output_file = "mel_spectrograms.csv"

# Extract Mel spectrograms for all audio files
mel_spectrograms = []
labels = []
for category in ["REAL", "FAKE"]:
    category_path = os.path.join(audio_dir, category)
    if os.path.exists(category_path):
        for file_name in os.listdir(category_path):
            if file_name.endswith(".flac"):
                file_path = os.path.join(category_path, file_name)
                mel_spec = extract_melspectrogram(file_path)
                mel_spectrograms.append(mel_spec.tolist())  # Convert to list for CSV storage
                labels.append(category)
                print(f"Processed: {file_name}")

# Create a DataFrame and save it to CSV
df = pd.DataFrame({
    "mel_spec": mel_spectrograms,
    "label": labels
})

df.to_csv(output_file, index=False)

print(f"Mel spectrograms extracted and saved to {output_file}")
