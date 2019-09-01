#%% Importando módulos
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
from medImUtils import misc, filters, changeFormat, info

#%% 1 - Funções temporais

s1 = misc.tfunc(f=1)
s2 = misc.tfunc(f=3)
s3 = misc.tfunc(f=5)
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
X = filters.FT(s1+s2+s3,t,10)
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('TF')
plt.stem(np.arange(0,11),np.abs(X))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

#%% 2 - Pulso quadrado
squarePulse = imageio.imread(r'ImagensAulas\PulsoQuadrado1.pgm')
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Square Pulse')
plt.imshow(squarePulse,cmap='gray',vmin = 0, vmax=255)
plt.axis('off')
plt.show()

#%% 3 - FFT2 Pulso quadrado
fftsquarePulse,fftshiftsquarePulse,absfftsquarePulse = filters.imageFFT(squarePulse)
logFFT = np.log(1+fftshiftsquarePulse)
logFFTABS = np.abs(logFFT)
logFFTABS = changeFormat.im2uint8(changeFormat.imNormalize(logFFTABS))
info.showImageStyle(1,3,{'fftsquarePulse': changeFormat.im2uint8(changeFormat.imNormalize(np.abs(fftsquarePulse))),'absfftsquarePulse':absfftsquarePulse,'logFFTABS':logFFTABS},['fftsquarePulse','absfftsquarePulse','LogfftsquarePulse'])

#%% 4 - Filtro passa baixas de 10%
H = np.zeros(squarePulse.shape, dtype = int)
centerX, centerY = int((squarePulse.shape[0])/2),int((squarePulse.shape[1])/2)
filter_length = centerX*0.5
for i in range (H.shape[0]):
    for j in range (H.shape[1]):
        if ((centerX - i)**2 + (centerY - j)**2)**0.5 <= filter_length:
            H[i,j] = 1
 
Ffiltrado = absfftsquarePulse*H
info.showImageStyle(1,3,{'absfftsquarePulse':absfftsquarePulse,'H':changeFormat.im2uint8(H),'Ffiltrado':Ffiltrado },['absfftsquarePulse','H','Ffiltrado'])
Ffiltrado = fftshiftsquarePulse*H

#%% 5 - Transformada inversa de Fourier
Ifft,IfftABS = filters.imageIFFT(Ffiltrado, show = True)

#%% 5a - Filtragem de mamo e stent
mamo = imageio.imread(r'ImagensAulas\Mamography.pgm')
stent = imageio.imread('ImagensAulas\Stent.pgm')
filtroMamo = filters.ffilter(mamo,1,0.2)
filtroStent = filters.ffilter(stent,0.8,0.3)

mamoFFT,mamoFFTshift,absMamo = filters.imageFFT(mamo,False)
mamoIFFT,mamoIFFTabs = filters.imageIFFT(mamoFFTshift*filtroMamo,False)
info.showImageStyle(1,2,{'mamo':mamo,'mamoIFFTshift':np.uint8(255*mamoIFFTabs)},['mamo','mamo filtered'])

stentFFT,stentFFTshift,absStent = filters.imageFFT(stent,False)
stentIFFT,stentIFFTabs = filters.imageIFFT(stentFFTshift*filtroStent,False)
info.showImageStyle(1,2,{'stent':stent,'stentIFFTshift':np.uint8(255*stentIFFTabs)},['stent','stent filtered'])




