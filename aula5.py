#%% Importando módulos
import numpy as np
import matplotlib.pyplot as plt
import imageio
from medImUtils import imUtils
import os

#%% 1 - Funções temporais
def tfunc(f):
    import numpy as np
    t = np.arange(0,10.01,0.01)
    return np.sin(2*np.pi*f*t)

s1 = tfunc(1)
s2 = tfunc(3)
s3 = tfunc(5)
t = np.arange(0,10.01,0.01)

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Sinais')
plt.subplot(411)
plt.plot(t,s1)
plt.title('S1')
plt.xlim((0,10))
plt.subplot(412)
plt.plot(t,s2)
plt.title('S2')
plt.xlim((0,10))
plt.subplot(413)
plt.plot(t,s3)
plt.title('S3')
plt.xlim((0,10))
plt.subplot(414)
plt.plot(t,s1+s2+s3)
plt.title('S sum')
plt.xlim((0,10))
plt.show()


#%% 1b - Tranformada de Fourier
def TF(signal,time):
    X = np.zeros(11,dtype = complex) # matriz de zeros do tipo complexo para dar bom a FFT
    for i in range(11):
        X[i] = np.sum(signal*np.exp((-1j)*2*np.pi*i*time))
    return X

X = TF(s1+s2+s3,t)
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('TF')
plt.stem(np.arange(0,11),np.abs(X))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

#%% 2 - Pulso quadrado
cf = os.getcwd() # current folder
squarePulse_path = os.path.join(cf,'ImagensAulas','PulsoQuadrado1.pgm') # endereco da imagem
squarePulse = imageio.imread(squarePulse_path)

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Square Pulse')
plt.imshow(squarePulse,cmap='gray',vmin = 0, vmax=255)
plt.axis('off')
plt.show()

#%% 3 - FFT2 Pulso quadrado
squarePulseFFT = np.fft.fft2(squarePulse)
squarePulseABS = imUtils.im2uint8(np.abs(squarePulseFFT))
squarePulseFFTshift = np.fft.fftshift(squarePulseFFT)
squarePulseABS2 = imUtils.im2uint8(np.abs(squarePulseFFTshift))
logFFT = np.log(1+squarePulseFFTshift)
logFFTABS = imUtils.im2uint8(np.abs(logFFT))

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Square Pulse FFT')
plt.subplot(131)
plt.imshow(squarePulseABS)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(squarePulseABS2)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(133)
plt.imshow(logFFTABS)
plt.set_cmap('gray')
plt.axis('off')

plt.show()

#%% 4 - Filtro passa baixas de 10%
H = np.zeros(squarePulse.shape, dtype = int)
centerX, centerY = int((squarePulse.shape[0])/2),int((squarePulse.shape[1])/2)
filter_length = centerX*0.5
for i in range (H.shape[0]):
    for j in range (H.shape[1]):
        if ((centerX - i)**2 + (centerY - j)**2)**0.5 <= filter_length:
            H[i,j] = 1
 
Ffiltrado = squarePulseFFTshift*H
FfiltradoABS = imUtils.im2uint8(np.abs(Ffiltrado))

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Square Pulse FFT')
plt.subplot(131)
plt.imshow(squarePulseABS2)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(H)
plt.set_cmap('gray')
plt.axis('off')  
plt.subplot(133)
plt.imshow(FfiltradoABS)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

#%% 5 - Transformada inversa de Fourier
Ifft = np.fft.ifftshift(Ffiltrado)
Ifft = np.fft.ifft2(Ifft)
IfftABS = imUtils.im2uint8(np.abs(Ifft))

Ifft2 = np.fft.ifftshift(squarePulseFFTshift)
Ifft2 = np.fft.ifft2(Ifft2)
IfftABS2 = imUtils.im2uint8(np.abs(Ifft2))

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Square Pulse FFT')
plt.subplot(121)
plt.imshow(IfftABS)
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(IfftABS2)
plt.set_cmap('gray')
plt.axis('off')
plt.show()

#%% 5a - Filtragem de mamo e stent
mamo_path = os.path.join(cf,'ImagensAulas','Mamography.pgm') # endereco da imagem
mamo = imageio.imread(mamo_path)

stent_path = os.path.join(cf,'ImagensAulas','Stent.pgm') # endereco da imagem
stent = imageio.imread('G:\Meu Drive\CODIGOS\PYTHON\ImagensMedicas\ImagensAulas\Stent.pgm')

filtroMamo = imUtils.ffilter(mamo,0.1)
filtroStent = imUtils.ffilter(stent,0.1)

mamoFFT,mamoFFTshift,absMamo = imUtils.imageFFT(mamo,False)
mamoIFFT,mamoIFFTshift,mamoIFFTabs = imUtils.imageIFFT(mamoFFTshift*filtroMamo,False)
imUtils.showImageStyle(1,2,{'mamo':mamo,'mamoIFFTshift':mamoIFFTshift},['mamo','mamo filtered'])

stentFFT,stentFFTshift,absStent = imUtils.imageFFT(stent,False)
stentIFFT,stentIFFTshift,stentIFFTabs = imUtils.imageIFFT(stentFFTshift*filtroStent,False)
imUtils.showImageStyle(1,2,{'stent':stent,'stentIFFTshift':stentIFFTshift},['stent','stent filtered']) # para plotar do meu jeito




