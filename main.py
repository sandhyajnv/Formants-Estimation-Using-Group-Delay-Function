import librosa 
import numpy as np 
from scipy.fft import fft, fftfreq,fftshift
import scipy.signal 
import matplotlib.pyplot as plt
import math
import pandas as pd
from google.colab import drive
drive.mount("/content/drive")

from scipy.signal import find_peaks

from scipy.signal import savgol_filter

speechFile,sr=librosa.load("/content/drive/MyDrive/speech_files/oak.wav",sr=None)
print(len(speechFile))
x=np.linspace(0,len(speechFile)/sr,len(speechFile),endpoint=False)
winsize=0.003 # seconds
hopl=0.015 # seconds
framelength=int(0.03*sr)
hoplength=int(0.015*sr)
nfft=512
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.plot(x,speechFile)

frames=librosa.util.frame(speechFile,frame_length=framelength, hop_length=hoplength, axis=0)
hw=scipy.signal.get_window("hamming",nfft)
dftx=fftfreq(nfft,1/sr)

def padding(nfft,framelength,frames):
  
  zero=np.zeros(nfft)
  point=[]
  for i in range(len(frames)):
    zero=np.zeros(nfft)
    zero[:framelength]=frames[i]
    point.append(zero)

  return(np.array(point))



def groupDelay(framesp,hw):
  y=[]
  for frames in framesp:
    y1=[]
    for index,value in enumerate(frames):
      y1.append(index*value)
    y.append(y1)
  

  y=np.array(y)
  x=framesp
  dftxg=[]
  dftyg=[]
  for i in range(len(x)):
    dftxg.append((fft(np.multiply(hw,x[i]))))
    dftyg.append((fft(np.multiply(hw,y[i]))))
  
  dftxg=np.array(dftxg)
  dftyg=np.array(dftyg)
  
  

  sum=[]
  for i in range(len(dftxg)):
    x=np.multiply(dftxg.real,dftyg.real)
    y=np.multiply(dftxg.imag,dftyg.imag)
    

  return(x+y)
    

  
def plot_timeDomainframe(frames,framelength,i,sr):
  plt.title("frame no {}".format(i))
  
  plt.plot(np.linspace(0,framelength/sr,framelength,endpoint=False),frames[i])
  plt.xlabel("time(s)")
  plt.ylabel("amplitude")
  plt.title("time domain plot of frame {}".format(i))
  plt.show()
  
    
def hwfft(framesp,hw):
  dftyh=[]
  for i in range(len(framesp)):
    dftyh.append(abs(fft(np.multiply(hw,framesp[i]))))
  return(np.array(dftyh))
    #dfty.append(abs(fft(framesp[i])))



def peak_picking(t,dftx):
  frames_S=[]
  for frames in t:
    frames_S.append(savgol_filter(frames[:nfft//2],51,3))
  
  formant=[]
  peak=[]
  for frame in frames_S:
    peaks, _=find_peaks(frame)
    
    formant.append([dftx[:nfft//2][i] for i in peaks])
    peak.append(peaks)

  return(formant,frames_S,peak)

def FFT_plotting(x,y,i,nfft):
  plt.plot(x,np.log10(abs(y[i][:nfft//2])))
  plt.xlabel("Frequency(Hz)")
  plt.ylabel("Magnitude ")
  plt.show()


def gd_plot(x,y,i,nfft):
  plt.title("group delay")
  plt.xlabel("Frequency(Hz)")
  plt.ylabel("time(s)")
  plt.plot(x,y[i][:nfft//2])
  plt.show()
  

def peak_plot(dftx,frames_S,nfft,i):
  plt.xlabel("frequency")
  plt.ylabel("peaks ")
  plt.plot(dftx[peak[i]][:nfft//2],frames_S[i][peak[i]][:nfft//2],"x")
  plt.plot(dftx[:nfft//2],frames_S[i][:nfft//2])
  plt.show()

framesp=padding(nfft,framelength,frames)
fft_frames=hwfft(framesp,hw)

t=groupDelay(framesp,hw)
formants,frames_S,peak=peak_picking(t,dftx)
fn=65 # frame number
FFT_plotting(dftx[:nfft//2],fft_frames,fn,nfft)

plot_timeDomainframe(frames,framelength,fn,sr) # plotting time domain frame

#gd_plot(dftx[:nfft//2],frames_S,fn,nfft)

peak_plot(dftx,frames_S,nfft,fn) # plot of frames with corresponding peaks marked


