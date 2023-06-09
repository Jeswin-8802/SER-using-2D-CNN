
# Speech Emotion Recognition Using 2D CNN

A ML model that is capable in predicting the type of human emotiom from speech using the 2D CNN algorithm.

<br>

## <b> How it Works </b>
---

A 2D CNN model works well with detecting patterns in images and we do the same here. An audio sample from which multiple audio and acoustic features will be extracted and converted into a large 2D array which corresponds to an image which can be fed to the model.

#### <b> The following datasets have been considered: </b>
- RAVDESS
- CREMA
- TESS
- SURREY

They can be downloaded by executing the following script: https://github.com/Nathaniel538/SER-using-2D-CNN/blob/main/scripts/download_audio_files.py 

#### <b> The following features were selected to train the model: </b>
- Chroma STFT
- Mel Spectrogram
- MFCC
- Spectral Contrast
- Tonnetz

For more detailed information on the uage of the features in feature extraction refer: https://github.com/Nathaniel538/SER-using-2D-CNN/blob/main/exploration_and_extraction/feature_extraction_v2.ipynb

<br>

![Alt text](/images/image-3.png)

<i> The concatenation of multiple features vertically results in a large 2D array which when converted into an image gives </i>

<br>

## <b> Model Constraction and Training </b>
---

The constructed model has the following model summary:

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 max_pooling2d (MaxPooling2D  (None, 53, 80, 1)        0
 )

 dropout (Dropout)           (None, 53, 80, 1)         0

 conv2d (Conv2D)             (None, 51, 78, 64)        640

 max_pooling2d_1 (MaxPooling  (None, 25, 39, 64)       0
 2D)

 dropout_1 (Dropout)         (None, 25, 39, 64)        0

 conv2d_1 (Conv2D)           (None, 23, 37, 64)        36928

 flatten (Flatten)           (None, 54464)             0

 dense (Dense)               (None, 64)                3485760

 dropout_2 (Dropout)         (None, 64)                0

 dense_1 (Dense)             (None, 32)                2080

 dense_2 (Dense)             (None, 8)                 264

=================================================================
Total params: 3,525,672
Trainable params: 3,525,672
Non-trainable params: 0
_________________________________________________________________
```

which when represented visually looks like:

![Alt text](/images/image-4.png)

<br>

The model is trained with a total of 36,000 audio samples whch includes variations of the multiple audio files from the dataset for a total of 100 epochs.
It can be trained using the following python script: https://github.com/Nathaniel538/SER-using-2D-CNN/blob/main/scripts/train_model.py

<br>

The trained model has an accuracy of 80.99135756492615 % on test data.
The accuracy and loss over the duration of training the model:
![Alt text](/images/image-5.png)

<br>

And the overall classification report of the trained model on the test data:

```
 precision    recall  f1-score   support

       angry       0.91      0.88      0.89      1430
        calm       0.70      0.83      0.76       138
     disgust       0.80      0.80      0.80      1379
        fear       0.78      0.79      0.78      1409
       happy       0.84      0.73      0.78      1407
     neutral       0.80      0.78      0.79      1175
         sad       0.72      0.83      0.77      1396
    surprise       0.94      0.94      0.94       462

    accuracy                           0.81      8796
   macro avg       0.81      0.82      0.81      8796
weighted avg       0.81      0.81      0.81      8796
```

<br>

## <b> File Tree </b>

```
|   .gitignore
|   Pipfile
|   Pipfile.lock
|   README.md
|   
+---data
|   |   accuracy_tracker.md
|   |   data_path.csv
|   |   data_x.npz
|   |   data_y.npz
|   |   lstm.pkl
|   |   lstm_v1.pkl
|   |   
|   +---crema          
|   +---ravdess       
|   +---surrey     
|   \---tess
|                   
+---exploration_and_extraction
|       exploratory_analysis.ipynb
|       feature_extraction_v2.ipynb
|       model_development_v1.ipynb
|       model_development_v2.ipynb
|       
+---lib
|   |   custom_util.py
|   |   util_for_2d_features.py
|   |   __init__.py
|           
\---scripts
        create_data_path_csv.py
        extract_features.py
        model_test_v1.py
        model_test_v2.py
        train_model.py
        download_audio_files.py
        download_features.py
```

The entire project can be worked from start to finish within `\scripts`
