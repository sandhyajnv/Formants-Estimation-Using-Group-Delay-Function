 Formants-Estimation-Using-Group-Delay-Function 
 
This project finds the formants frequencies of a speech signal.

. In this project User provides a speech file(wav file)

. Native sampling rate is preserved for the speech file.

.Choose appropriate win- size and hop length while making frames of the given speech file.


INSTALLATION INSTRUCTION:

Download the main code.  This code uses signal processing and speech processing libraries as scipy and librosa respectively.

Installation of libraries:
pip install --user scipy,

pip install librosa

How to use:
Download the wav file from example data file and run the main code by providing path of your speech file and to get formants corresponding to each frame print formants.
  
  
 peaks for frame=65
![peak](https://user-images.githubusercontent.com/105709848/203570290-f4af001f-9f8c-4e35-9c80-e37673edb0a1.PNG)
