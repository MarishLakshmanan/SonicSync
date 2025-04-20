import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import joblib
from ensemble_model import EnsembleModel

# Set up paths
DATASET_PATH = './archive/Data/genres_original'

# Define function to extract features from audio
def extract_features(file_path, n_mfcc=40):
    """
    Extracts Mel Frequency Cepstral Coefficients (MFCC) from an audio file.
    """
    audio, sr = librosa.load(file_path, sr=None, duration=30)  # Load audio with 30 sec duration
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Create dataset of features and labels
def prepare_dataset(dataset_path):
    X = []  # Features
    y = []  # Labels

    # Map genres to indices
    genres = os.listdir(dataset_path)
    genre_map = {genre: idx for idx, genre in enumerate(genres)}

    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)

        for file_name in os.listdir(genre_path):
            filepath = os.path.join(genre_path, file_name)

            try:
                # Load the audio file
                signal, sr = librosa.load(filepath, sr=22050)

                # Correctly calculate MFCCs
                mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
                mfcc = np.mean(mfcc.T, axis=0)  # Mean over time axis

                # Append features and labels
                X.append(mfcc)
                y.append(genre_map[genre])

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    return np.array(X), np.array(y)


# Prepare dataset
print("Extracting features and labels...")
X, y = prepare_dataset(DATASET_PATH)

# Normalize data
X = X / np.max(X)

# Reshape data for CNN (add a channel dimension)
X = X.reshape(X.shape[0], 1, X.shape[1], 1)

# Encode labels
genres = sorted(set(y))
label_to_index = {genre: idx for idx, genre in enumerate(genres)}
y_encoded = np.array([label_to_index[label] for label in y])
y_one_hot = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# CNN model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Input(shape=(1, 13, 1)),  # Updated input shape (1, n_mfcc, 1)
        Conv2D(32, (1, 3), activation='relu', padding='same'),  # Adjusted kernel size for width
        MaxPooling2D((1, 2)),  # Pooling on height and width
        Conv2D(64, (1, 3), activation='relu', padding='same'),  # Adjusted kernel size
        MaxPooling2D((1, 2)),  # Pooling
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Final layer for classification
    ])
    return model

# Compile CNN
cnn_model = build_cnn(input_shape=X_train.shape[1:], num_classes=y_one_hot.shape[1])
cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduce learning rate by half
    patience=3,  # Wait for 3 epochs of no improvement
    min_lr=1e-6  # Minimum learning rate
)

history = cnn_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=[reduce_lr]
)

# Extract features for RF using CNN
print("Extracting features using CNN...")
feature_extractor = Sequential(cnn_model.layers[:-1])  # Use CNN layers except the final Dense
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# cnn_model.save('cnn_model.h5')
print("CNN model saved as cnn_model.h5")

# Train Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_features, np.argmax(y_train, axis=1))

# joblib.dump(rf_model, 'rf_model.pkl')
print("Random Forest model saved as rf_model.pkl")

# Evaluate model
y_pred_rf = rf_model.predict(X_test_features)
print("Random Forest Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred_rf))
print(f"Accuracy: {accuracy_score(np.argmax(y_test, axis=1), y_pred_rf):.2f}")

# Map genres to indices
genres = os.listdir(DATASET_PATH)
genre_map = {genre: idx for idx, genre in enumerate(genres)}

# Reverse mapping from index to genre
index_to_genre = {idx: genre for genre, idx in genre_map.items()}

def predict_genre(file_path):
    # Extract features from the new song
    features = extract_features(file_path, n_mfcc=13)
    features = np.mean(features.T, axis=0)  # Average across the time axis

    # Preprocess features (normalize and reshape)
    features = features / np.max(features)  # Normalize
    features = features.reshape(1, 1, 13, 1)  # Reshape for CNN input

    # Extract features using the CNN feature extractor
    features_extracted = feature_extractor.predict(features)

    # Predict genre using the Random Forest model
    predicted_genre_index = rf_model.predict(features_extracted)[0]

    # Convert index to genre name
    predicted_genre_name = index_to_genre[predicted_genre_index]
    return predicted_genre_name

# Example usage
new_song_path = "./temp_audio/song.mp3"
print(f"The predicted genre is: {predict_genre(new_song_path)}")

# Create an instance of the EnsembleModel
ensemble_model = EnsembleModel(cnn_model=cnn_model, rf_model=rf_model)

# Train the ensemble model
ensemble_model.fit(X_train, np.argmax(y_train, axis=1))
# Save the ensemble model
joblib.dump(ensemble_model, 'ensemble_genre_model.pkl')
print("Ensemble model saved as ensemble_genre_model.pkl")
