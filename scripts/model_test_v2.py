import joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing
import plotext as plt
import librosa
import sys
# sys.tracebacklimit = 0
import os
import re
if os.path.abspath('../lib') not in sys.path:
    sys.path.insert(0, os.path.abspath('../lib'))
import util_for_2d_features as util
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category = DeprecationWarning)



def main():
    if len(sys.argv) <= 1:
        raise Exception("No Audio file given...")
    path_to_sound_file = sys.argv[1]
    checkFileValidity(path_to_sound_file)
    test_data = np.array([get_features(path_to_sound_file)])
    test_data = np.expand_dims(test_data, axis = 3)
    print(test_data.shape)
    predicted_data = model.predict(test_data)
    print(predicted_data)
    percentages = preprocessing.minmax_scale(predicted_data, feature_range=(0, 100), axis=1, copy=True).round(0)
    print(percentages)

    getProbableEmotion(percentages[0])


def get_features(path_to_sound_file):
    y, sr = librosa.load(path_to_sound_file)
    features = util.get_features(y, sr)
    return features



def getProbableEmotion(predicted_data):
    plt.bar(['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], predicted_data)
    plt.xlabel('Emotions')
    plt.ylabel('Predicted Emotion')
    plt.title('Probabity of Possible Emotions')
    plt.show()



def checkFileValidity(path_to_sound_file):
    pattern = r'\.mp3|\.wav|\.flac|\.m4a'
    if not re.search(pattern, path_to_sound_file):
        raise Exception("Invalid sound file format...")
    if not os.path.isfile(path_to_sound_file):
        raise Exception("File not found")



if __name__ == '__main__':
    model_file_location = os.path.join(os.path.abspath('..'), 'data', 'lstm.pkl')
    history = joblib.load(model_file_location)
    global model
    model = history.model

    main()