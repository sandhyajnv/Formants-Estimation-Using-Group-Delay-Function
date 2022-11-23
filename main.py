import librosa 
import numpy as np 
from scipy.fft import fft, fftfreq,fftshift
import scipy.signal 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks   # for finding peaks

from scipy.signal import savgol_filter



#loading speech file 
speechFile,sr=librosa.load("C:\\Users\\Aditya choudhary\\OneDrive\\Desktop\\pp\\speech_files-20221121T152622Z-001\\speech_files\\oak.wav",sr=None)
 #   print(len(speechFile)) #  len of speech file 
x=np.linspace(0,len(speechFile)/sr,len(speechFile),endpoint=False) #time in sec.
winsize=0.003 # seconds  (one frame length)
hopl=0.015 # seconds (hoplength)
framelength=int(0.03*sr)
hoplength=int(0.015*sr)
nfft=512 # npoint fft
plt.xlabel("Time")
plt.ylabel("Amplitude")

# plot betwwen time and amplitude of speech file
plt.plot(x,speechFile)
plt.show()
# for dividing sppech signal into different frames with given frame length and hop length
frames=librosa.util.frame(speechFile,frame_length=framelength, hop_length=hoplength, axis=0)
hw=scipy.signal.get_window("hamming",nfft)


dftyh=[]
dfty=[]
dftx=fftfreq(nfft,1/sr)
zeron=np.zeros(nfft) # array of nfft length
point=[]
for i in range(len(frames)):
  zeron=np.zeros(nfft)
  zeron[:framelength]=frames[i] # adding extra zeros because of no. of samples = 30* sr
  point.append(zeron)

framesp=np.array(point)  # new frame samples # len(framessp) =199 (for oak. file)

 

for i in range(len(framesp)):
  dftyh.append(abs(fft(np.multiply(hw,framesp[i]))))  # multiply by haming window for smmothing
  dfty.append(abs(fft(framesp[i])))



dfty=np.array(dfty)


# for dividing signal ibnto frames with given hop length and frame length
frames=librosa.util.frame(speechFile,frame_length=framelength, hop_length=hoplength, axis=0)
hw=scipy.signal.get_window("hamming",nfft) #geting singmnal by hamming window
dftx=fftfreq(nfft,1/sr)

def padding(nfft,framelength,frames): # for padding zeros for exta position grater than smaples 
  
  zero=np.zeros(nfft)
  point=[]
  for i in range(len(frames)):
    zero=np.zeros(nfft)
    zero[:framelength]=frames[i]
    point.append(zero)

  return(np.array(point))



def groupDelay(framesp,hw): # group delay function
  y=[]
  for frames in framesp:
    y1=[]
    for index,value in enumerate(frames):
      y1.append(index*value)      # for y = nx[n]
    y.append(y1)
    
  y=np.array(y)
  x=framesp
  dftxg=[]   # dftx for group delay
  dftyg=[]    # dfty for group delay
  for i in range(len(x)):
    dftxg.append((fft(np.multiply(hw,x[i])))) # applying hamming window
    dftyg.append((fft(np.multiply(hw,y[i]))))  # hamming window
  
  dftxg=np.array(dftxg)   # changing in array
  dftyg=np.array(dftyg)  # changing in array
  
  

  for i in range(len(dftxg)):
    x=np.multiply(dftxg.real,dftyg.real)
    y=np.multiply(dftxg.imag,dftyg.imag) 
  
  return (x+y)  # matrix after applying group delay function
    

  
def timeDomainframe(i,frames,framelength,sr):
  plt.title("frame no {}".format(i))
  
  plt.plot(np.linspace(0,framelength/sr,framelength,endpoint=False),frames[i])
  plt.xlabel("time(s)")
  plt.ylabel("amplitude")
  plt.title("time domain plot of frame {}".format(i))
  plt.show()
  


def hwfft(framesp,hw):
  dftyh=[]       #fft after applying hamming window
  for i in range(len(framesp)):
    dftyh.append(abs(fft(np.multiply(hw,framesp[i]))))
  return(np.array(dftyh))
    #dfty.append(abs(fft(framesp[i])))



    
framesp=padding(nfft,framelength,frames)

x=hwfft(framesp,hw)

t=groupDelay(framesp,hw)


def FFT_plotting(x,y,i,nfft):
  plt.title("frame no {}".format(i))
  plt.plot(x,np.log10(abs(y[i][:nfft//2])))
  plt.xlabel("Frequency(Hz)")
  plt.ylabel("Magnitude ")
  plt.show()


def gd_plot(x,y,i,nfft):
  plt.title("group delay for frame {}".format(i))
  plt.xlabel("Frequency(Hz)")
  plt.ylabel("time(s)")
  plt.plot(x,y[i][:nfft//2])
  plt.show()


def energy_plot(I): # function for energy plot
    ts = []
    for i in I:
      ns =0
      for j in i :
        j = max(j,0.00001)
        ns +=np.log(j*j)
      ts.append(ns)
    plt.xlabel("frames")
    plt.ylabel("energy level") 
    plt.title("Energy level plot")
    plt.plot(ts)
    plt.show()

def peak_picking(t,dftx):
  frames_S=[]
  for frames in t:
    frames_S.append(savgol_filter(frames[:nfft//2],47,3))
  
  formant=[]
  for frame in frames_S:
    peaks, _=find_peaks(frame)
    
    formant.append([dftx[:nfft//2][i] for i in peaks])

  return(formant,frames_S)


frame_number = 65

#n_peak = 65
energy_plot(framesp) 


framesp=padding(nfft,framelength,frames)
fft_frames=hwfft(framesp,hw)

t=groupDelay(framesp,hw)
formants,frames_S=peak_picking(t,dftx) # formants contains all the formants frequencies corresponding to each frame

FFT_plotting(dftx[:nfft//2],fft_frames,frame_number,nfft)


timeDomainframe(frame_number,frames,framelength,sr)  # time domain plot for given frame number    

gd_plot(dftx[:nfft//2],t,frame_number,nfft) #plotting group delay function against frequency
gd_plot(dftx[:nfft//2],frames_S,frame_number,nfft)# plotting smoothed group delay function against frequency






