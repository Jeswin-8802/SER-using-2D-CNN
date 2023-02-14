import numpy as np
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Audio


def get_audio(df, emotion):
    path = np.array(df.Path[df.Emotions == emotion])[0]
    print(path)
    audio = Audio(path)
    display(audio)


def get_audio_by_index(df, num):
    path = df.Path[num]
    print(path)
    audio = Audio(path)
    display(audio)


def print_emotion_count(df):
    emotion_count = df['Emotions'].value_counts()

    plt.rcParams['figure.dpi'] = 77
    sns.set(style='whitegrid')
    plt.subplots(figsize=(12, 4))
    sns.barplot(x=emotion_count.index, y=emotion_count, palette='rocket')
    plt.xlabel('Emotions', size=18, color='#4f4e4e')
    plt.ylabel('Count', size=18, color='#4f4e4e')
    plt.title('Count of Emotions', size=18, color='#4f4e4e')
    plt.xticks(size=14, color='#4f4e4e')
    plt.yticks(size=14, color='#4f4e4e')

    y_cord = emotion_count.max() // 30
    for i in range(len(emotion_count)):
        plt.text(x=i, y=y_cord, s=emotion_count[i], color='white',
                 fontsize=18, horizontalalignment='center')

    sns.despine(left=True)


def create_waveplot(df, emotion):
    audio_array, sampling_rate = librosa.load(
        np.array(df.Path[df.Emotions == emotion])[0])
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(audio_array, sr=sampling_rate)
    plt.title('Audio Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')


def display_audio(data):
    sample_rate = 22050  # default sampling rate passed
    if (type(data) == ''.__class__ and data.endswith('.wav')):
        data, sample_rate = librosa.load(data)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=data, sr=sample_rate)
    audio = Audio(data, rate=sample_rate)
    display(audio)


def create_spectrogram(df, emotion):
    data, sampling_rate = librosa.load(
        np.array(df.Path[df.Emotions == emotion])[0])
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    plt.title('Spectrogram for audio with {} emotion'.format(emotion), size=15)
    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
    plt.colorbar()


def noise(path):
    data, sample_rate = librosa.load(path)
    noise_amp = 0.025*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def stretch(path, rate=0.9):
    data, sample_rate = librosa.load(path)
    return librosa.effects.time_stretch(data, rate)


def shift(path):
    data, sample_rate = librosa.load(path)
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def pitch(path, pitch_factor=0.9):
    data, sample_rate = librosa.load(path)
    return librosa.effects.pitch_shift(data, sample_rate, pitch_factor)


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(path)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(path)
    data_stretch_pitch = pitch(path)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    return result


def extract_features_and_compress_to_1d(data, sample_rate = 16825):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally
    # print(zcr.shape)

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally
    # print(chroma_stft.shape)

    # poly features
    p10 = np.mean(librosa.feature.poly_features(S=stft, order=9).T, axis=0)
    result = np.hstack((result, p10))  # stacking horizontally
    # print(p10.shape)

    # MFCC of order 3
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=3).T, axis=0)
    result = np.hstack((result, mfcc_delta2))  # stacking horizontally
    # print(mfcc_delta2.shape)

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally
    # print(rms.shape)

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(
        y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally
    # print(mel.shape)

    # spectral centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate))
    result = np.hstack((result, spectral_centroid)) # stacking horizontally
    # print(spectral_centroid)

    # spectral rolloff
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(data+0.01, sr=sample_rate))
    result = np.hstack((result, spectral_rolloff)) # stacking horizontally
    # print(spectral_rolloff)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast)) # stacking horizontally
    # print(contrast.shape)

    # tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz)) # stacking horizontally
    # print(tonnetz.shape)

    return result
