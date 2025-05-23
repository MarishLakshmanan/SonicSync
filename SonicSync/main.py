import json
import os
import re
from base64 import b64encode
from enum import Enum
from io import BytesIO
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator, root_validator, confloat, conlist, conint
from typing import Union, Optional
import soundfile as sf
from scipy.fft import fft
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
import librosa
from ensemble_model import EnsembleModel
import dill as pickle
from tensorflow.keras.models import load_model
import torch.nn.functional as F
import torch
from genre_classifier import GenreClassifier
import joblib
import inspect


from autoeq.constants import DEFAULT_BASS_BOOST_GAIN, DEFAULT_BASS_BOOST_FC, DEFAULT_BASS_BOOST_Q, \
    DEFAULT_TREBLE_BOOST_GAIN, DEFAULT_TREBLE_BOOST_FC, DEFAULT_TREBLE_BOOST_Q, DEFAULT_TILT, DEFAULT_FS, \
    DEFAULT_MAX_GAIN, DEFAULT_SMOOTHING_WINDOW_SIZE, DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE, DEFAULT_TREBLE_F_LOWER, \
    DEFAULT_TREBLE_F_UPPER, DEFAULT_TREBLE_GAIN_K, DEFAULT_PHASE, DEFAULT_PREAMP, DEFAULT_F_RES, \
    PEQ_CONFIGS, DEFAULT_BIT_DEPTH, DEFAULT_STEP, DEFAULT_SOUND_SIGNATURE_SMOOTHING_WINDOW_SIZE, DEFAULT_MAX_SLOPE
from autoeq.frequency_response import FrequencyResponse

ROOT_DIR = Path().resolve()

app = FastAPI()

# ensemble_config = {
#     'cnn_model_path': 'cnn_model.h5',
#     'rf_model_path': 'rf_model.pkl'
# }

# joblib.dump(ensemble_config, 'ensemble_config.pkl')

index_to_genre = {0: 'blues', 1: 'metal', 2: 'classical', 3: 'country', 4: 'rock', 5: 'hiphop', 6: 'reggae', 7: 'pop', 8: 'jazz', 9: 'disco'}


with open(ROOT_DIR.joinpath('data/entries.json')) as fh:
    entries = json.load(fh)

with open(ROOT_DIR.joinpath('data/measurements.json')) as fh:
    measurements = json.load(fh)

with open(ROOT_DIR.joinpath('data/targets.json')) as fh:
    targets = json.load(fh)


@app.get('/entries')
def get_entries():
    return entries




@app.get('/targets')
def get_targets():
    return [{key: target[key] for key in ['label', 'compatible', 'recommended', 'bassBoost']} for target in targets]


@app.get('/playlist')
def get_playlist():
    playlist = []
    extension_pattern = re.compile(r'\.(wav|flac|map3|aac|ogg|opus)$', flags=re.IGNORECASE)
    full_pattern = re.compile(r'^.*\.(wav|flac|map3|aac|ogg|opus)$', flags=re.IGNORECASE)
    for fp in ROOT_DIR.joinpath('data/audio').glob('*'):
        if re.match(full_pattern, str(fp.name)):
            playlist.append({
                'name': re.sub(extension_pattern, '', fp.name),
                'url': f'audio/{fp.name}'
            })
    return playlist


class MeasurementData(BaseModel):
    frequency: list[float]
    raw: list[float]


class Optimizer(BaseModel):
    min_f: Optional[float]
    max_f: Optional[float]
    max_time: Optional[confloat(ge=0.0, le=0.5)]
    min_change_rate: Optional[float]
    min_std: Optional[float]
    target_loss: Optional[float]


class FilterTypeEnum(str, Enum):
    LOW_SHELF = 'LOW_SHELF'
    HIGH_SHELF = 'HIGH_SHELF'
    PEAKING = 'PEAKING'


class Filter(BaseModel):
    type: Optional[str]
    fc: Optional[float]
    min_fc: Optional[float]
    max_fc: Optional[float]
    q: Optional[float]
    min_q: Optional[float]
    max_q: Optional[float]
    gain: Optional[float]
    min_gain: Optional[float]
    max_gain: Optional[float]


class PEQConfig(BaseModel):
    optimizer: Optional[Optimizer]
    filter_defaults: Optional[Filter]
    filters: conlist(Filter, min_items=1)


class BitDepthEnum(int, Enum):
    PCM_16 = 16
    PCM_32 = 32


class PhaseEnum(str, Enum):
    minimum = 'minimum'
    linear = 'linear'


class ResponseRequirements(BaseModel):
    fr_f_step = DEFAULT_STEP
    fr_fields: Optional[list[str]]
    base64fp16 = False


class EqualizeRequest(BaseModel):
    measurement: Optional[MeasurementData]
    name: Optional[str]
    source: Optional[str]
    rig: Optional[str]
    target: Optional[Union[str, MeasurementData]]
    bass_boost_gain = DEFAULT_BASS_BOOST_GAIN
    bass_boost_fc = DEFAULT_BASS_BOOST_FC
    bass_boost_q = DEFAULT_BASS_BOOST_Q
    treble_boost_gain = DEFAULT_TREBLE_BOOST_GAIN
    treble_boost_fc = DEFAULT_TREBLE_BOOST_FC
    treble_boost_q = DEFAULT_TREBLE_BOOST_Q
    tilt = DEFAULT_TILT
    fs: Optional[int] = DEFAULT_FS
    bit_depth: Optional[BitDepthEnum] = DEFAULT_BIT_DEPTH
    f_res: Optional[float] = DEFAULT_F_RES
    phase: Optional[PhaseEnum] = DEFAULT_PHASE
    sound_signature: Optional[MeasurementData]
    sound_signature_smoothing_window_size: Optional[float] = DEFAULT_SOUND_SIGNATURE_SMOOTHING_WINDOW_SIZE
    max_gain = DEFAULT_MAX_GAIN
    max_slope = DEFAULT_MAX_SLOPE
    window_size = DEFAULT_SMOOTHING_WINDOW_SIZE
    treble_window_size = DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE
    treble_f_lower = DEFAULT_TREBLE_F_LOWER
    treble_f_upper = DEFAULT_TREBLE_F_UPPER
    treble_gain_k = DEFAULT_TREBLE_GAIN_K
    parametric_eq = False
    parametric_eq_config: Optional[Union[str, PEQConfig, list[Union[str, PEQConfig]]]] = '8_PEAKING_WITH_SHELVES'
    fixed_band_eq = False
    fixed_band_eq_config: Optional[Union[str, PEQConfig]] = '10_BAND_GRAPHIC_EQ'
    graphic_eq = False
    convolution_eq = False
    preamp = DEFAULT_PREAMP
    response: Optional[ResponseRequirements]

    @root_validator
    def only_one_eq_type(cls, values):
        assert values.get('measurement') or (values.get('name') and values.get('source') and values.get('rig')), 'Measurement is required'
        keys = ['parametric_eq', 'fixed_band_eq', 'equalizer_apo_graphic_eq', 'convolution_eq']
        assert len([key for key in keys if values.get(key)]) < 2, 'Only one equalizer type is allowed'
        return values

    @validator('parametric_eq_config')
    def parametric_eq_config_name(cls, v):
        if type(v) == str:
            assert v in PEQ_CONFIGS, f'Unknown parametric eq config name "{v}"'
        if type(v) == list:
            for config in v:
                if type(config) == str:
                    assert config in PEQ_CONFIGS, f'Unknown parametric eq config name "{config}"'
        return v

    @validator('fixed_band_eq_config')
    def fixed_band_eq_config_name(cls, v):
        if type(v) == str:
            assert v in PEQ_CONFIGS, f'Unknown fixed band eq config name "{v}"'
        return v


def magnitude_response(x, fs):
    """Calculates frequency magnitude response

    Args:
        x: Audio data
        fs: Sampling rate

    Returns:
        - **f:** Frequencies
        - **X:** Magnitudes
    """
    nfft = len(x)
    df = fs / nfft
    f = np.arange(0, fs - df, df)
    y = fft(x)
    y_mag = 20 * np.log10(np.abs(y))
    return f[0:int(np.ceil(nfft/2))], y_mag[0:int(np.ceil(nfft/2))]

def extract_features(file_path, n_mfcc=40):
    """
    Extracts Mel Frequency Cepstral Coefficients (MFCC) from an audio file.
    """
    audio, sr = librosa.load(file_path, sr=None, duration=30)  # Load audio with 30 sec duration
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs

async def load_my_model(file_location):
    
    # Load the ensemble model
    loaded_ensemble_model = joblib.load('ensemble_genre_model.pkl')
    print("Ensemble model loaded successfully!")

    # new_song_path = "./temp_audio/song.mp3"
    new_features = extract_features(file_location, n_mfcc=13)

    # Calculate the mean across the time axis
    new_features = np.mean(new_features.T, axis=0)

    # Normalize the features
    new_features = new_features / np.max(new_features)

    # Reshape for CNN input
    new_features = new_features.reshape(1, 1, 13, 1)  # Reshape to (batch_size, channels, height, width)

    predicted_genre_index = loaded_ensemble_model.predict(new_features)[0]

    # Convert index to genre name
    predicted_genre_name = index_to_genre[predicted_genre_index]
    return predicted_genre_name

def load_spectogram(file_path,target_shape=(150,150)):
    try:
        # if filename.endswith('.wav'):
        # file_path = os.path.join(class_dir, filename)
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
    
async def load_my_model2(file_location):
    print(inspect.getfile(GenreClassifier))
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    gc = GenreClassifier(num_classes=10)
    model = joblib.load("./model.pkl")

    # Preprocess the audio file
    mel_spectrograms = load_spectogram(file_location, target_shape=(150, 150))
    if mel_spectrograms is None:
        return "Error processing audio file"

    

    # Predict genre
    predictions = []
    for mel_spectrogram in mel_spectrograms:
        mel_spectrogram_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0)
        mel_spectrogram_tensor = mel_spectrogram_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            output = model(mel_spectrogram_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_index = probabilities.argmax(dim=1).item()
            predictions.append(predicted_index)

    # Determine the most common predicted genre
    # print(predictions)
    predicted_genre_index = max(set(predictions), key=predictions.count)
    print(classes[predicted_genre_index])
    return classes[predicted_genre_index]

def preprocess_audio(file_path):
    y, sr= librosa.load(file_path, sr=None)
    return y

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_location = f"temp_audio/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Make prediction
        model = await load_my_model(file_location)
        # print(model)
        model2 = await load_my_model2(file_location)
        result = {"model1":model,"model2":model2}

        # Clean up (optional: delete temp file)

        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post('/equalize')
def equalize(req: EqualizeRequest):
    try:
        if req.measurement:  # Custom measurement data provided
            fr = FrequencyResponse(name='fr', frequency=req.measurement.frequency, raw=req.measurement.raw)
        else:  # Named measurement
            measurement = measurements[req.name][req.source][req.rig]
            fr = FrequencyResponse(name='fr', frequency=measurement['frequency'], raw=measurement['raw'])

        if req.target is None:
            fr.smoothen(
                window_size=req.window_size,
                treble_window_size=req.treble_window_size,
                treble_f_lower=req.treble_f_lower,
                treble_f_upper=req.treble_f_upper
            )
            return {'fr': fr.to_dict()}
        elif type(req.target) == str:
            target = None
            for target_ in targets:
                if target_['label'] == req.target:
                    target = target_
            if target is None:
                raise ValueError(f'Unknown target {req.target}')
            target = FrequencyResponse(
                name='target', frequency=target['fr']['frequency'], raw=target['fr']['raw'])
        else:
            target = FrequencyResponse(
                name='target', frequency=req.target.frequency, raw=req.target.raw)

        if req.sound_signature is not None:
            sound_signature = FrequencyResponse(
                name='sound signature', frequency=req.sound_signature.frequency, raw=req.sound_signature.raw)
        else:
            sound_signature = None

        fr.process(
            target=target,
            min_mean_error=True,
            bass_boost_gain=req.bass_boost_gain,
            bass_boost_fc=req.bass_boost_fc,
            bass_boost_q=req.bass_boost_q,
            treble_boost_gain=req.treble_boost_gain,
            treble_boost_fc=req.treble_boost_fc,
            treble_boost_q=req.treble_boost_q,
            tilt=req.tilt,
            fs=req.fs,
            sound_signature=sound_signature,
            sound_signature_smoothing_window_size=req.sound_signature_smoothing_window_size,
            max_gain=req.max_gain,
            max_slope=req.max_slope,
            window_size=req.window_size,
            treble_window_size=req.treble_window_size,
            treble_f_lower=req.treble_f_lower,
            treble_f_upper=req.treble_f_upper,
            treble_gain_k=req.treble_gain_k)

        f_step = req.response.fr_f_step if req.response is not None else ResponseRequirements.fr_f_step
        fr_fields = req.response.fr_fields if req.response is not None else ResponseRequirements.fr_fields
        if fr_fields is None:
            fr_fields = list(fr.to_dict().keys())

        fr.interpolate(f_step=f_step)
        d = fr.to_dict()
        res = {'fr': {key: d[key] for key in fr_fields}}

        if req.parametric_eq:
            parametric_eq_config = req.parametric_eq_config
            if type(parametric_eq_config) != list:
                parametric_eq_config = [parametric_eq_config]
            parametric_eq_config = [
                PEQ_CONFIGS[config] if type(config) == str else config.dict() for config in parametric_eq_config
            ]
            # Limit maximum optimization time to 500 ms
            total_max_time = 0
            for config in parametric_eq_config:
                if 'optimizer' not in config:
                    config['optimizer'] = {'max_time': 0.5}
                elif (
                        'max_time' not in config['optimizer']
                        or config['optimizer']['max_time'] is None
                        or config['optimizer']['max_time'] > 0.5
                ):
                    config['optimizer']['max_time'] = 0.5
                total_max_time += config['optimizer']['max_time']
            max_time = 0.5 if total_max_time > 0.5 else None
            parametric_peqs = fr.optimize_parametric_eq(parametric_eq_config, req.fs, preamp=req.preamp, max_time=max_time)
            peq = parametric_peqs[0]
            peq.sort_filters()
            res['parametric_eq'] = peq.to_dict()
            res['parametric_eq']['preamp'] = peq.max_gain * -1 - 0.1
            peq_fr = FrequencyResponse(name='PEQ', frequency=peq.f, raw=peq.fr)
            peq_fr.interpolate(f_step=f_step)
            res['fr']['parametric_eq'] = peq_fr.raw.tolist()

        if req.fixed_band_eq:
            if type(req.fixed_band_eq_config) == str:
                fixed_band_eq_config = PEQ_CONFIGS[req.fixed_band_eq_config]
            else:
                fixed_band_eq_config = req.fixed_band_eq_config.dict()
            if 'optimizer' not in fixed_band_eq_config:
                fixed_band_eq_config['optimizer'] = {'max_time': 0.5}
            elif (
                    'max_time' not in fixed_band_eq_config['optimizer']
                    or fixed_band_eq_config['optimizer']['max_time'] is None
                    or fixed_band_eq_config['optimizer']['max_time'] > 0.5
            ):
                fixed_band_eq_config['optimizer']['max_time'] = 0.5

            fixed_band_peqs = fr.optimize_fixed_band_eq(
                fixed_band_eq_config, req.fs, preamp=req.preamp,
                gain_range=2.0 if len(fixed_band_eq_config['filters']) > 10 else None)
            fixed_band_peq = fixed_band_peqs[0]
            fixed_band_peq.sort_filters()
            res['fixed_band_eq'] = fixed_band_peq.to_dict()
            res['fixed_band_eq']['preamp'] = fixed_band_peq.max_gain * -1 - 0.1
            fbpeq_fr = FrequencyResponse('FBPEQ', frequency=fixed_band_peq.f, raw=fixed_band_peq.fr)
            fbpeq_fr.interpolate(f_step=f_step)
            res['fr']['fixed_band_eq'] = fbpeq_fr.raw.tolist()

        if req.graphic_eq:
            graphic_eq = fr.eqapo_graphic_eq(normalize=True, preamp=req.preamp)
            res['graphic_eq'] = graphic_eq

        if req.convolution_eq:
            bit_depth = req.bit_depth if req.bit_depth is not None else DEFAULT_BIT_DEPTH
            bit_depth = 'PCM_16' if bit_depth == BitDepthEnum.PCM_16 else 'PCM_32'
            f_res = req.f_res if req.f_res is not None else DEFAULT_F_RES
            preamp = req.preamp if req.preamp is not None else DEFAULT_PREAMP
            phase = req.phase if req.phase is not None else DEFAULT_PHASE
            if phase is None or phase == PhaseEnum.minimum or phase == 'minimum':
                fir = fr.minimum_phase_impulse_response(fs=req.fs, f_res=f_res, normalize=True, preamp=preamp).T
            elif phase == PhaseEnum.linear:
                fir = fr.linear_phase_impulse_response(fs=req.fs, f_res=f_res, normalize=True, preamp=preamp).T
            else:
                raise TypeError
            # Create WAV data buffer
            buf = BytesIO()
            sf.write(buf, fir, req.fs, bit_depth, format='WAV')
            buf.seek(0)
            res['fir'] = b64encode(buf.read())
            # Add FIR frequency response
            f, mag = magnitude_response(fir, req.fs)
            fir_fr = FrequencyResponse(name='FIR', frequency=f[1:], raw=mag[1:])
            fir_fr.interpolate(f_step=f_step)
            ix200 = np.argmin(np.abs(fr.frequency - 200))
            fir_fr.raw += np.mean(fr.equalization[ix200:] - fir_fr.raw[ix200:])
            res['fr']['convolution_eq'] = fir_fr.raw.tolist()

        base64fp16 = req.response.base64fp16 if req.response is not None else ResponseRequirements.base64fp16
        if base64fp16:
            res['fr'] = {key: b64encode(np.array(val, dtype='float16')) for key, val in res['fr'].items()}

        return res

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


app.mount('/audio', StaticFiles(directory=ROOT_DIR.joinpath('data/audio'), html=False), name='audio')
app.mount('/legal', StaticFiles(directory=ROOT_DIR.joinpath('data/legal'), html=True), name='legal')
if os.getenv('APP_ENV') == 'production':
    app.mount('/', StaticFiles(directory=ROOT_DIR.joinpath('ui/build'), html=True), name='build')



