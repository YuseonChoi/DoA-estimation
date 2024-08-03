# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:13:40 2018

@author: Juho Laukkanen,student number: 218886
"""

import os
import numpy as np
from scipy.signal import fftconvolve
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def read_audio(filename):
    _fs, _y = wav.read(filename)
    if _y.dtype == np.int16:
        _y = _y / 32768.0  # short int to float
    return _y, _fs

#(if) Signals are of different length, two choices either add zeros to the shorter or remove samples from the longer one:
#Remove samples from longer:
def prune(x,y):
    xLen = np.size(x)
    yLen = np.size(y)
    if xLen > yLen:
        x = x[0:yLen]
    else:
        y = y[0:xLen]
        
    return x,y

def srp_phat(s,fs,nFFT=None,center=None,d=None,azimuth_estm=None,mode=None):
    '''
    Applies Steered Power Response with phase transform algorithm
    Uses pyroomacoustics module
    
    Input params
    ------------
    s: numpy array
        -stacked microphone array signals 
        (NOTE: number of microphones is extracted from the size of input signal,
         since the input signal will be of size MxN where M is number of microphones
         and N is the length of the audio signal.)
    fs: int
        -Sampling frequency
    nfft: int
        -FFT size. Default 1024
    center: numpy array
        -Defines the center of the room. Default [0,0]
    d: int
        -Distance between microphones. Default 10cm.
    azimuth_estm: numpy array
        -Candidate azimuth estimates, representing location estimates of speakers.
         Default expects microphone to be in the middle of a table and speakers located around it.
         Assumes two speakers - [60,120]
    mode: str
        -Defines the microphone setup layout. Default mode = linear.
        mode = linear 
        mode = circular
    '''    
    if nFFT is None:
        nFFT = 1024
    if center is None:
        center = [0,0]
    if d is None:
        d = 0.1
    if azimuth_estm is None:
        azimuth_estm = [60,120]
        
    freq_bins = np.arange(30,330) #list of individual frequency bins used to run DoA
    M = s.shape[0] #number of microphones
    phi = 0 #assume angle between microphones is 0 (same y-axis)
    radius = d*M/(2*np.pi) #define radius for circular microphone layout
    c = 343.2 #speed of sound

    #Define Microphone array layout
    if mode is "circular":
        L = pra.circular_2D_array(center,M,phi,radius)
    if mode is None or "linear":
        L = pra.linear_2D_array(center,M,phi,d)

    nSrc = len(azimuth_estm) #number of speakers

    #STFT
    # s_FFT = np.array([np.transpose(pra.transform.stft.STFT(nFFT,nFFT//2,transform='numpy')) for s_ch in s])
    s_FFT = np.array([np.fft.rfft(s_ch,n=nFFT).T for s_ch in s])
    # s_FFT = np.array([pra.transform.stft.STFT(s_ch,nFFT,nFFT//2,transform=np.fft.rfft).T for s_ch in s]) #STFT for s1 and s2

    #SRP
    doa = pra.doa.srp.SRP(L,fs,nFFT,c,max_four=4,num_src=nSrc) #perform SRP approximation

    s_FFT = np.expand_dims(s_FFT,axis=-1)
    print(s_FFT.shape)
    #Apply SRP-PHAT
    doa.locate_sources(s_FFT,freq_bins=freq_bins)
    
    #PLOTTING
    doa.polar_plt_dirac()
    plt.title('SRP-PHAT')
    print('SRP-PHAT')
    print('Speakers at: ',np.sort(doa.azimuth_recon)/np.pi*180, 'degrees')
    plt.show()
    
def calculate_frequencies(yWin,fs):
    freq = np.fft.fftfreq(len(yWin),1.0/fs) #frequencies of the data frame
    freq = freq[1:]
    return freq
    
def calculate_energy(yWin,fs):
    ampl = np.abs(np.fft.fft(yWin))
    ampl = ampl[1:]    
    energy = ampl ** 2 
    return energy    
    
def detect(y,fs,thrs):
        '''
        Voice Activity Detection
        Detection is based within the ratio of max energy of the frame and speech band energy.
        
        Outputs 1's and 0's for each frame where speech was detected.
        '''
    
        frame_length = 0.02 #20ms
        frame_overlap = 0.01 #10ms overlap
        energy_thresh = thrs #threshold of detection
#        energy_thresh = 0.9 # threshold
                
        frame = int(fs * frame_length)
        frame_overlap = int(fs*frame_overlap)
        
        detected_windows = np.array([])  
        
        si = 0 #start index
        s_band = 300 #start band of voice frequency 
        e_band = 3400 #end band of voice frequency (wikipedia)
        
        while (si < (len(y)-frame)): #framewise processing
            ei = si+frame #frame end index
            
            if ei>=len(y):ei = len(y)-1 #if not at end
            yWin = y[si:ei] #take frame of data
            
            freqs = calculate_frequencies(yWin,fs) #calculate frames frequency components
            energy = calculate_energy(yWin,fs) #calculate frames energy 
            
            #Energy - Frequency 
            energy_freq = {}
            for (i,freq) in enumerate(freqs):
                if abs(freq) not in energy_freq:
                    energy_freq[abs(freq)] = energy[i]*2      
            
            sum_max_energy = sum(energy_freq.values())
            
            #Speech band energy
            sum_energy = 0
            for f in energy_freq.keys():
                if s_band < f < e_band:
                    sum_energy += energy_freq[f]
            
            #Calculate ratio
            ratio = sum_energy/sum_max_energy
            #Thresholding
            ratio = ratio>energy_thresh
            #[Frame, 1 or 0]
            detected_windows = np.append(detected_windows,[si,ratio])
            si += frame_overlap #move frame
            
        #Reshape
        detected_windows = detected_windows.reshape(len(detected_windows)//2,2)
        return detected_windows

def main():
    
    # print(os.getcwd())
    # audio_dir = "./data/mix/1mix/Normalize_False/Clean/high/"
    # audio_filename1 = 'hal_in_pure_24_4ch_48k_1.wav'
    # audio_filename2 = 'hal_in_pure_24_4ch_48k_2.wav'
    audio_dir = "./data/mix/2mix/Normalize_False/Clean/45_1+45_2/"
    audio_filename1 = 'hal_in_pure_24_4ch_48k_1.wav'
    audio_filename2 = 'hal_in_pure_24_4ch_48k_2.wav'
    s1,fs = read_audio(os.path.join(audio_dir,audio_filename1))
    s2,fs = read_audio(os.path.join(audio_dir,audio_filename2))
    
    sLen = np.size(s1)
    s = np.zeros((2,sLen),dtype=np.float32)

    s[0,:] = s1#[0*sLen:1*sLen]
    s[1,:] = s2#[0*sLen:1*sLen]

    #Where is the relative center inside the room (around which microphones are located)?
    #We know from schematic (https://www.researchgate.net/figure/The-layout-of-the-UEDIN-Instrumented-Meeting-Room-measurements-in-cm-Array_fig1_4208275)
    #That the room measurements are 650cm X and 490cm Y
    
    center = [0,0]
    d = 0.161 #distance between microphones (in meters)
    azimuth_estm = [30,60] #[60,120,240,320] #A subjective guess where the speaker could be
    mode = 'linear' #define microphone layout (circular or linear)
    nfft = 1024 #size of fft used
    
    srp_phat(s,fs,nfft,center,d,azimuth_estm,mode)
    
    threshold = 0.9 #threshold of detection (ratio of speech-max energy of a frame)
    detected_windows = detect(s,fs,0.9)

    print(detected_windows)
    return detected_windows

#IMPLEMENTATION OF ONLY VOICED FRAMES FOR SRP IS MISSING
main()


# y,fs = read_audio('test1.wav')
# detected_windows = detect(y,fs,0.9)


##One instance of AMI database arrays were that they weren't the same length
#if np.size(s1) != np.size(s2):
#    s1,s2 = prune(s1,s2) #make signals same length
    








