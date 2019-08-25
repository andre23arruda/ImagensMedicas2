#%% Importando módulos
import imageio
from medImUtils import imUtils
import os

#%% 1 - Funções de transferência - Filtro ideal
m,n,fc,ft1,show = 400,400,0.2,'LP',True
H_ideal = imUtils.idealfilter(m,n,fc,ft1,show)

#%% 2 - Funções de transferência - Filtro gaussiano
m,n,fc,ft1,show = 400,400,0.2,'LP',True
H_gaussian = imUtils.gaussianFilter2D(m,n,fc,ft1,show)

#%% 3 - Funções de transferência - Filtro butterworth
m,n,fc,ft1,show = 400,400,0.2,'LP',True
nPoles = 3
H_butter = imUtils.butterFilter2D(m,n,fc,nPoles,ft1,show)

#%% 1 - Filtragens Passa baixa
cf = os.getcwd() # current folder
mamo_path = os.path.join(cf,'ImagensAulas','Mamography.pgm') # endereco da imagem
mamo = imageio.imread(mamo_path)

mamoFFT,mamoFFTshift,mamoFFTabs = imUtils.imageFFT(mamo)
m,n,fc = mamoFFTshift.shape[0],mamoFFTshift.shape[1],0.2

H_ideal = imUtils.idealfilter(m,n,fc)
mamoFFT_filtered = mamoFFTshift * H_ideal
mamoIFFT,mamoIFFTshift,mamoIdeal = imUtils.imageIFFT(mamoFFT_filtered)

H_gaussian = imUtils.gaussianFilter2D(m,n,fc)
mamoFFT_filtered = mamoFFTshift * H_gaussian
mamoIFFT,mamoIFFTshift,mamoGaussian = imUtils.imageIFFT(mamoFFT_filtered)

H_butter = imUtils.butterFilter2D(m,n,fc,2)
mamoFFT_filtered = mamoFFTshift * H_butter
mamoIFFT,mamoIFFTshift,mamoButter = imUtils.imageIFFT(mamoFFT_filtered)

imUtils.showImageStyle(1,4,{'mamo':mamo,'mamoIdeal':mamoIdeal,'mamoGaussian':mamoGaussian,'mamoButter':mamoButter},['mamo','Ideal','Gaussian','Butter'])

#%% 2 - Filtragens Passa alta
mamo = imageio.imread(mamo_path)

mamoFFT,mamoFFTshift,mamoFFTabs = imUtils.imageFFT(mamo)
m,n,fc = mamoFFTshift.shape[0],mamoFFTshift.shape[1],0.2

H_ideal = imUtils.idealfilter(m,n,fc,filterType = 'HP')
mamoFFT_filtered = mamoFFTshift * H_ideal
mamoIFFT,mamoIFFTshift,mamoIdeal = imUtils.imageIFFT(mamoFFT_filtered)

H_gaussian = imUtils.gaussianFilter2D(m,n,fc,filterType = 'HP')
mamoFFT_filtered = mamoFFTshift * H_gaussian
mamoIFFT,mamoIFFTshift,mamoGaussian = imUtils.imageIFFT(mamoFFT_filtered)

H_butter = imUtils.butterFilter2D(m,n,fc,2,filterType = 'HP')
mamoFFT_filtered = mamoFFTshift * H_butter
mamoIFFT,mamoIFFTshift,mamoButter = imUtils.imageIFFT(mamoFFT_filtered)

imUtils.showImageStyle(1,4,{'mamo':mamo,'mamoIdeal':mamoIdeal,'mamoGaussian':mamoGaussian,'mamoButter':mamoButter},['mamo','Ideal','Gaussian','Butter'])


