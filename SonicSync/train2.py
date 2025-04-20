import os
import librosa
import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.model_selection import ParameterGrid
# from torchsummary import summary
from torchinfo import summary
warnings.filterwarnings("ignore")
from genre_classifier import GenreClassifier
gpu_index = os.environ.get('CUDA_VISIBLE_DEVICES')
if gpu_index:
    print(f"Using SLURM-assigned GPU(s): {gpu_index}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# import pickle
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report

DATASET_PATH = './archive/Data/genres_original'

def save_model_as_pkl(model, filepath="model.pkl"):
    joblib.dump(model, filepath)
    print(f"Model saved as {filepath}")


def load_model_from_pkl(filepath="model.pkl"):
    loaded_model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return loaded_model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, params, filepath, save_best=True):
    model.to(device)
    best_val_acc = 0  # Track the best validation accuracy
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == y_batch).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss = {train_loss / len(train_loader):.4f}, "
              f"Train Acc = {train_correct / len(train_loader.dataset):.4f}, "
              f"Val Loss = {val_loss / len(val_loader):.4f}, "
              f"Val Acc = {val_acc:.4f},")

        # Save the model if it achieves the best validation accuracy
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            # save_model(model, params, epoch + 1, val_acc)
            save_model_as_pkl(model, filepath)

    return best_val_acc  # Return the best validation accuracy achieved

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

best_acc = 0
best_params = None

params = {
    'epochs': 20,
    'lstm_units': 128,
    'dropout_rate': 0.5,
    'learning_rate': 0.0005,
    'batch_size': 64
}

def load_spectogram(class_dir,filename,target_shape=(150,150)):
    try:
        # if filename.endswith('.wav'):
        file_path = os.path.join(class_dir, filename)
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        if len(audio_data) == 0:
            print(f"Skipping empty or corrupted file: {file_path}")
            return None
        chunk_duration = 4
        overlap_duration = 2
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
        spectrograms = []
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            chunk = audio_data[start:end]
            if len(chunk) < chunk_samples:
                print(f"Skipping incomplete chunk from file: {file_path}")
                continue
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            if mel_spectrogram.shape[1] < target_shape[1]:
                mel_spectrogram = np.pad(mel_spectrogram,
                                        ((0, 0), (0, target_shape[1] - mel_spectrogram.shape[1])),
                                        mode='constant')
            if mel_spectrogram.shape[0] < target_shape[0]:
                mel_spectrogram = np.pad(mel_spectrogram,
                                        ((0, target_shape[0] - mel_spectrogram.shape[0]), (0, 0)),
                                        mode='constant')
            mel_spectrogram = mel_spectrogram[:target_shape[0], :target_shape[1]]
            mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
            spectrograms.append(mel_spectrogram)

        return spectrograms
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_mel_spectrogram(data_dir, classes, target_shape=(150,150)):
    data = []
    labels = []
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            spectrograms = load_spectogram(class_dir,filename,target_shape)
            if spectrograms is None:
              continue
            for mel_spectrogram in spectrograms:
              data.append(mel_spectrogram)
              labels.append(i_class)

    return np.array(data), np.array(labels)

def load_data():
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    data, labels = load_mel_spectrogram(DATASET_PATH, classes)
    return data, labels

class GenreDataset():
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X, y = load_data()
num_classes = 10
print(f"X shape : {X.shape}, \n Y shape : {y.shape}")
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = GenreDataset(X_train, y_train)
test_dataset = GenreDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

model = GenreClassifier(num_classes=num_classes, lstm_units=params['lstm_units'], dropout_rate=params['dropout_rate'])
print("Model summary : ")
print(model)
print("Proper model summary : ")
summary(model, input_size=(16, 1, 150, 150))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=params['epochs'], params=params, filepath='model.pkl')

print(f"\n***********\nParams: {params}, Val Accuracy: {val_acc:.4f}\n************")

val_loss, val_correct = 0, 0
all_preds = []
all_labels = []
model = load_model_from_pkl(filepath="model.pkl")
model.eval()
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        val_loss += loss.item()
        val_correct += (outputs.argmax(1) == y_batch).sum().item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

val_acc = val_correct / len(val_loader.dataset)
print(f"\n\n\n\nValidation Accuracy: {val_acc:.4f}")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))
