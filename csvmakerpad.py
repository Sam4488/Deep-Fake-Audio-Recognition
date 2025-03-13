import librosa
import numpy as np
import pandas as pd
import os

# Fixed duration in seconds
FIXED_DURATION = 4  # 4 seconds

# Function to load, pad/crop, and extract features
def extract_features(file_path, label, fixed_duration=FIXED_DURATION):
    y, sr = librosa.load(file_path, sr=None)  # Load at native sampling rate
    target_length = sr * fixed_duration  # Desired number of samples

    # Crop or pad the audio to exactly 4 seconds
    if len(y) > target_length:
        y = y[:target_length]  # Truncate if too long
    else:
        y = np.pad(y, (0, max(0, target_length - len(y))), mode='constant')  # Pad with zeros

    # Extract features
    features = {
        "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y))
    }

    # Extract MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i+1}"] = np.mean(mfccs[i])

    # Add label
    features["LABEL"] = label
    return features

# Path to the main audio directory
audio_dir = "E:\BIJI\DATASET2"
output_file = "extracted_features2_VERIFY.csv"

# Extract features for all audio files
feature_list = []
for category in ["REAL", "FAKE"]:
    category_path = os.path.join(audio_dir, category)
    if os.path.exists(category_path):
        for file_name in os.listdir(category_path):
            if file_name.endswith(".flac"):
                file_path = os.path.join(category_path, file_name)
                feature_list.append(extract_features(file_path, category))
                print(f"Processed: {file_name}")

# Convert to DataFrame
df_features = pd.DataFrame(feature_list)

# Normalize column-wise (excluding LABEL)
numeric_columns = df_features.columns.difference(["LABEL"])
df_features[numeric_columns] = (df_features[numeric_columns] - df_features[numeric_columns].mean()) / df_features[numeric_columns].std()

# Save to CSV
df_features.to_csv(output_file, index=False)

print(f"Features extracted from 4-second audio clips, normalized, and saved to {output_file}")
