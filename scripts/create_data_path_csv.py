import pandas as pd
import os
import re

def get_paths_to_datasets():
    paths = {}
    base_dir = os.path.abspath('..')
    Ravdess = os.path.join(base_dir, 'data', 'ravdess')
    Crema = os.path.join(base_dir, 'data', 'crema', 'AudioWAV')
    Savee = os.path.join(base_dir, 'data', 'surrey', 'ALL')    
    Tess = os.path.join(base_dir, 'data', 'tess', 'TESS Toronto emotional speech set data')
    paths['Ravdess'] = Ravdess
    paths['Crema'] = Crema
    paths['Savee'] = Savee
    paths['Tess'] = Tess
    print(paths)
    return paths

def get_crema_df(path):
    crema_directory_list = os.listdir(path)
    crema_emotions = {
        'NEU': 'neutral',
        'HAP': 'happy',
        'SAD': 'sad',
        'ANG': 'angry',
        'FEA': 'fear',
        'DIS': 'disgust'
    }
    file_emotion = []
    file_path = []
    for file in crema_directory_list:
        # storing file paths
        file_path.append(os.path.join(path, file))
        emotion = file.split('_')[2]
        if emotion in crema_emotions:
            file_emotion.append(crema_emotions[emotion])
        else:
            file_emotion.append('Unknown')
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)
    return Crema_df

def get_tess_df(path):
    tess_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []
    for dir in tess_directory_list:
        directories = os.listdir(os.path.join(path, dir))
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]   
            if part == 'ps':
                file_emotion.append('surprise')
            elif part == 'happy(1)':
                file_emotion.append('happy')
            else:
                file_emotion.append(part)
            file_path.append(os.path.join(path, dir, file))
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    return Tess_df

def get_ravdess_df(path):
    ravdess_directory_list = os.listdir(path)
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        actor = os.listdir(os.path.join(path, dir))
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            file_emotion.append(int(part[2]))
            file_path.append(os.path.join(path, dir, file))
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    Ravdess_df.Emotions.replace(
        {
            1: 'neutral',
            2: 'calm',
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'fear',
            7: 'disgust',
            8: 'surprise'
        },
        inplace=True)
    return Ravdess_df


def get_savee_df(path):
    savee_directory_list = os.listdir(path)
    savee_emotions = {
        'n': 'neutral',
        'h': 'happy',
        'sa': 'sad',
        'su': 'surprise',
        'a': 'angry',
        'f': 'fear',
        'd': 'disgust'
    }
    file_emotion = []
    file_path = []
    for file in savee_directory_list:
        file_path.append(os.path.join(path, file))
        file_emotion.append(savee_emotions[re.search(r'\w\w_(\w+)\d\d', file)[1]])
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)
    Savee_df

def main():
    paths = get_paths_to_datasets()
    Crema_df = get_crema_df(paths['Crema'])
    Tess_df = get_tess_df(paths['Tess'])
    Ravdess_df = get_ravdess_df(paths['Ravdess'])
    Savee_df = get_savee_df(paths['Savee'])
    data_path = pd.concat([Crema_df, Tess_df, Ravdess_df, Savee_df], axis=0)
    to_store_at = os.path.join(os.path.abspath('..'), 'data', 'data_path.csv')
    data_path.to_csv(to_store_at, index=False)
    print('data_path.csv created in', to_store_at)

if __name__ == '__main__':
    main()
