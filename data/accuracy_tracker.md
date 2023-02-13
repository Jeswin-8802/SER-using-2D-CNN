# Model Accuracy

<!-- Use (Alt + Shift + F) to Reformat the table with indents -->
<!-- Use (Ctrl + Shift + V) for preview -->

| Features Used                                                                       | Sample Rate | Epochs | Accuracy             |
| :---------------------------------------------------------------------------------- | :---------- | :----- | :------------------- |
| ZCR, Chroma_stft, MFCC, RMS, MelSpectogram                                          | default     | 50     | 60.4911208152771 %   |
| ZCR, Chroma_stft, MFCC, RMS, MelSpectogram, Tonnetz                                 | default     | 50     | 61.173003911972046 % |
| RMS, MelSpectogram, spectral centroid, spectral rolloff, spectral contrast, Tonnetz | 16825       | 100    | 64.20926451683044 %  |
| Chroma_stft, poly_features, MFCC, MelSpectogram                                     | default     | 10     | 50.46029090881348 %  |
| Chroma_stft, poly_features, MFCC, MelSpectogram                                     | default     | 10     | 53.26883792877197 %  |


```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d_42 (Conv1D)          (None, 177, 64)           384       
                                                                 
 max_pooling1d_42 (MaxPoolin  (None, 89, 64)           0         
 g1D)                                                            
                                                                 
 conv1d_43 (Conv1D)          (None, 89, 128)           41088     
                                                                 
 max_pooling1d_43 (MaxPoolin  (None, 45, 128)          0         
 g1D)                                                            
                                                                 
 dropout_25 (Dropout)        (None, 45, 128)           0         
                                                                 
 flatten_13 (Flatten)        (None, 5760)              0         
                                                                 
 dense_26 (Dense)            (None, 32)                184352    
                                                                 
 dropout_26 (Dropout)        (None, 32)                0         
                                                                 
 dense_27 (Dense)            (None, 8)                 264       
                                                                 
=================================================================

Epoch 1/35
458/458 [==============================] - 20s 42ms/step - loss: 1.6078 - accuracy: 0.3594 - val_loss: 1.3769 - val_accuracy: 0.4676 - lr: 0.0010
Epoch 2/35
458/458 [==============================] - 21s 46ms/step - loss: 1.4003 - accuracy: 0.4402 - val_loss: 1.2741 - val_accuracy: 0.4902 - lr: 0.0010
Epoch 3/35
458/458 [==============================] - 19s 42ms/step - loss: 1.3306 - accuracy: 0.4675 - val_loss: 1.2183 - val_accuracy: 0.5167 - lr: 0.0010
Epoch 4/35
458/458 [==============================] - 20s 43ms/step - loss: 1.2952 - accuracy: 0.4778 - val_loss: 1.1945 - val_accuracy: 0.5210 - lr: 0.0010
Epoch 5/35
458/458 [==============================] - 21s 45ms/step - loss: 1.2758 - accuracy: 0.4857 - val_loss: 1.2040 - val_accuracy: 0.5177 - lr: 0.0010
Epoch 6/35
458/458 [==============================] - 20s 44ms/step - loss: 1.2483 - accuracy: 0.4906 - val_loss: 1.1763 - val_accuracy: 0.5317 - lr: 0.0010
Epoch 7/35
458/458 [==============================] - 21s 45ms/step - loss: 1.2319 - accuracy: 0.4978 - val_loss: 1.1567 - val_accuracy: 0.5406 - lr: 0.0010
Epoch 8/35
458/458 [==============================] - 20s 43ms/step - loss: 1.2138 - accuracy: 0.5052 - val_loss: 1.1598 - val_accuracy: 0.5356 - lr: 0.0010
Epoch 9/35
458/458 [==============================] - 19s 42ms/step - loss: 1.2027 - accuracy: 0.5127 - val_loss: 1.1301 - val_accuracy: 0.5451 - lr: 0.0010
```