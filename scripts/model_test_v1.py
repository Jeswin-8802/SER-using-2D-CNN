import joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing
import plotext as plt
import sys
# sys.tracebacklimit = 0
import os
import re
import custom_util as util
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category = DeprecationWarning)



def main():
    if len(sys.argv) <= 1:
        raise Exception("No Audio file given...")
    path_to_sound_file = sys.argv[1]
    checkFileValidity(path_to_sound_file)
    features = util.get_features(path_to_sound_file)
    Features = getSelectedFeatures(features) 
    test_data = Features.values
    np.expand_dims(test_data, axis = 2)
    predicted_data = model.predict(test_data)
    print(predicted_data)
    percentages = preprocessing.minmax_scale(predicted_data, feature_range=(0, 100), axis=0, copy=True).round(0)
    print(percentages)

    getProbableEmotion(percentages[0])



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



def getSelectedFeatures(features):
    Features = pd.DataFrame(features)
    features_to_choose = [1,2,3,5,9] # enter here
    present_features = [i for i in range(10)]
    to_remove = sorted([i for i in present_features if i not in features_to_choose], reverse=True)
    track_features = [
        ['ZCR', 1],
        ['Chroma_stft', 12],
        ['poly_features', 10],
        ['MFCC', 20],
        ['RMS', 1],
        ['MelSpectogram', 128],
        ['spectral centroid', 1],
        ['spectral rolloff', 1],
        ['spectral contrast', 7],
        ['tonnetz', 6]
    ]
    column_tracker, temp = [0], 0
    for data in track_features:
        temp += data[1]
        column_tracker.append(temp)
    for num in to_remove:
        Features = Features.drop(
            Features.iloc[:, column_tracker[num]:column_tracker[num + 1]], axis=1)
    
    return Features



if __name__ == '__main__':
    model_file_location = os.path.join(os.path.abspath('..'), 'data', 'lstm_v1.pkl')
    history = joblib.load(model_file_location, )
    global model
    model = history.model

    main()