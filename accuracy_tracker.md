# Model Accuracy

<!-- Use (Alt + Shift + F) to Reformat the table with indents -->

| Features Used                                                                       | Sample Rate | Epochs | Accuracy             |
| :---------------------------------------------------------------------------------- | :---------- | :----- | :------------------- |
| ZCR, Chroma_stft, MFCC, RMS, MelSpectogram                                          | default     | 50     | 60.4911208152771 %   |
| ZCR, Chroma_stft, MFCC, RMS, MelSpectogram, Tonnetz                                 | default     | 50     | 61.173003911972046 % |
| RMS, MelSpectogram, spectral centroid, spectral rolloff, spectral contrast, Tonnetz | 16825       | 100    | 64.20926451683044 %  |
