import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
import custom_util as util



def get_chroma_stft(S, chroma):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].label_outer()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])



def get_mel_spectrogram(S, sr):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=sr,
                             fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')



def get_mfcc_vs_mel_spectrogram(S, mfccs):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                   x_axis='time', y_axis='mel', fmax=8000,
                                   ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')



def get_spectral_contrast(S, contrast):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img1 = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[0])
    fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()
    img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
    fig.colorbar(img2, ax=[ax[1]])
    ax[1].set(ylabel='Frequency bands', title='Spectral contrast')



def get_tonnetz(y, sr, tonnetz):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img1 = librosa.display.specshow(tonnetz,
                                    y_axis='tonnetz', x_axis='time', ax=ax[0])
    ax[0].set(title='Tonal Centroids (Tonnetz)')
    ax[0].label_outer()
    img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),
                                    y_axis='chroma', x_axis='time', ax=ax[1])
    ax[1].set(title='Chroma')
    fig.colorbar(img1, ax=[ax[0]])
    fig.colorbar(img2, ax=[ax[1]])




def get_features(y, sr):
    S = librosa.feature.melspectrogram(y = y, sr = sr)
    mel_spectrogram = np.copy(S)

    S = np.abs(librosa.stft(y))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
    mfcc = librosa.feature.delta(mfcc, order=2)

    S = np.abs(librosa.stft(y))
    spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    y = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # mel_spectrogram shape ==> (128, 99)
    # chroma shape ==> (12, 99)
    # mfcc shape ==> (40, 99)
    # spectral_contrast shape ==> (7, 99)
    # tonnetz shape ==> (6, 99)
    temp = np.mean(mel_spectrogram[:4], axis=0)
    for i in range(4, len(mel_spectrogram) - 4, 4):
        temp = np.vstack((temp, np.mean(mel_spectrogram[i:i+4], axis=0)))

    result = temp
    result = np.vstack((result, chroma))
    result = np.vstack((result, mfcc))
    result = np.vstack((result, spectral_contrast))
    result = np.vstack((result, tonnetz))

    return cv2.resize(result, (160, result.shape[0]), interpolation= cv2.INTER_LINEAR)




def get_all_features_with_variations(y, sr, result):
    result = np.vstack((result, [get_features(y, sr)]))

    # noise
    noise_amp = 0.025 * np.random.uniform() * np.amax(y)
    y = y + noise_amp * np.random.normal(size = y.shape[0])
    result = np.vstack((result, [get_features(y, sr)]))

    # pitch
    y = librosa.effects.time_stretch(librosa.effects.pitch_shift(y, sr, 0.9), 0.7)
    result = np.vstack((result, [get_features(y, sr)]))

    return result