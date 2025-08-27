

## 🔊 Deep-Fake Audio Recognition

### 📝 Project Overview
This project focuses on detecting **deepfake audio**—synthetic voice recordings created using advanced AI techniques like speech synthesis, voice conversion, and replay attacks. With the growing threat of voice manipulation in sensitive fields like **banking, customer service, and law enforcement**, it’s crucial to develop robust methods to differentiate real voices from fake ones.

Our approach combines powerful **deep learning models** with smart **feature extraction** and **transfer learning** techniques to spot fake audio with high accuracy.

---

### 🧠 Models Used
To tackle deepfake audio detection, we experimented with a diverse set of powerful models—each bringing unique strengths to the table:

🔁 **BiLSTM (Bidirectional LSTM)**
Captures temporal patterns in audio sequences from both past and future directions for better context understanding.

🔊 **SE-Enhanced 1D-CNN for Time Series Classification**
Combines Squeeze-and-Excitation blocks with 1D CNNs to boost important audio features while suppressing noise.

🧱 **Custom CNN Architecture**
Tailor-made convolutional layers designed to extract deep, hierarchical features from spectrograms and MFCCs.

🌲 **Random Forest Classifier**
A robust ensemble-based method used for baseline comparisons and feature importance analysis.

🌐 **WIREnet**
A deep learning model tailored for audio spoofing detection, known for its effectiveness on ASVspoof datasets.

🌀 **ResNet FC (Fully Convolutional Residual Network)**
Utilizes residual connections for deep feature learning, reducing vanishing gradient issues.

🎯 **ResCNN-Attention-BiGRU**
A hybrid model that merges CNNs with Attention-powered Bidirectional GRUs for both spatial and sequential feature focus.

🧮 **SVM (Support Vector Machine)**
Classic yet effective, SVMs are used here for linear and non-linear classification on extracted feature vectors.

---

### 🎵 Audio Features Extracted
To detect fakes effectively, we extract the following features:
- **Spectrograms**
- **MFCCs** (Mel-Frequency Cepstral Coefficients)
- **Chroma STFT**
- **Spectral Centroid**
- **Spectral Bandwidth**
- **Zero Crossing Rate**

All features are **normalized** to maintain consistency across samples.

---

### 📊 Dataset Used

#### 📌 ASVspoof 2019 (Main Dataset)
A benchmark dataset for automatic speaker verification spoofing detection.

- **Logical Access (LA)**: Includes synthetic voices via TTS and voice conversion.
- **Physical Access (PA)**: Includes replayed voices in real-world scenarios.

**Data Split**:
- Training: 17,615 samples  
- Validation: 5,413 samples  
- Testing: 3,212 samples  

#### 📁 Additional Datasets:
- **RFP Dataset**
- **FakeAVCeleb Dataset**

---

### ⚙️ Methodology

#### 1. **Data Preprocessing**
- Clean audio
- Extract relevant features
- Normalize the data

#### 2. **Model Training**
- Models built using **PyTorch**
- Trained on the extracted features
- Evaluation via **cross-validation**

#### 3. **Evaluation Metrics**
- **F1-Score**
- **Tandem Decision Cost Function (t-DCF)**

---

### 🏆 Results
Our models showed **high accuracy on the ASVspoof 2019 dataset**, proving their ability to detect deepfake audio in controlled environments. However, challenges remain when generalizing to external datasets—a common issue in real-world deepfake detection.

---

### 🚀 Future Enhancements

- **Domain Adaptation**: Boost generalization using noise injection, pitch shifting, and speed variation.
- **Multimodal Learning**: Integrate **audio with video or text** for better accuracy.
- **Self-Supervised Learning**: Use **Wav2Vec 2.0** and **HuBERT** for stronger feature learning.
- **Graph Neural Networks**: Capture relationships between audio features more effectively.
- **Quantum Computing**: Explore faster computation for deepfake detection models.

---

### 👨‍💻 Contributors
- Kavya Verma  
- Ojas Kulkarni  
- Sagnik Samanta  
- **Kabir Gulati**  
- Divyanshi Mittal  
- Muzaffar Ahmad Dar  
- C. L. Biji  

---
