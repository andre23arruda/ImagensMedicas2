#%% Importando módulos
import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage import exposure
from ImageMedicalLib import info,misc

#%% 1- Lendo imagens
Mamo = imageio.imread(r'ImagensAulas\Mamography.pgm')
Stent = imageio.imread(r'ImagensAulas\Stent.pgm')

#%% 2 - Exibindo Mamo
plt.figure()
plt.imshow(Mamo)
plt.set_cmap('gray')
plt.title('Mamo')
plt.show()

#%% 3 - 'for' passando por cada pixel
M,N = Mamo.shape
Mamo_test = Mamo.copy().astype(float) # criando uma cópia para ficar zero bala
for indexrow,frow in enumerate(Mamo_test):
    for indexcolumn,fcolumn in enumerate(frow): 
        Mamo_test[indexrow,indexcolumn] = 1+ indexcolumn + (Mamo_test.shape[1])*indexrow# colocando o numero da posição do elemento atual

#%% 4 - Negativo com 'for'
M,N = Mamo.shape
Mamo_negativo = Mamo.copy().astype(float) # criando uma cópia para ficar zero bala
for i in range(M): # Pegando a linha toda de mamo
    for j in range(N): # pegando cada elemento da linha atual
        Mamo_negativo[i,j] = np.abs(Mamo_negativo[i,j] - 255)

plt.figure()
plt.imshow(Mamo_negativo)
plt.set_cmap('gray')
plt.title('Mamo negative')
plt.show()

#%% 5 - Negativo direto
Mamo_negativo_direto = np.uint8(np.abs(Mamo.astype(float)-255))
plt.figure()
plt.imshow(Mamo_negativo_direto)
plt.set_cmap('gray')
plt.title('Mamo negative')
plt.show()

#%% 6 - Função histograma
h = info.doHistogram(Stent,show = True)

#%% 7 - Aumentando brilho
StentBrilho = misc.imBrightness(Stent,50)
hist  = info.doHistogram(StentBrilho)
plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('Aumentando brilho')
plt.subplot(121)
plt.stem(hist[1,:])

plt.subplot(122)
plt.imshow(StentBrilho)
plt.set_cmap('gray')
plt.title('Stent + 50')
plt.show()

#%% 8 - Ajuste de contraste
StentContraste = misc.imAdjust(StentBrilho,0.2,0.7,gamma = 1)
hist  = info.doHistogram(StentContraste)
plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('Ajuste de contraste')
plt.subplot(121)
plt.stem(hist[1,:])

plt.subplot(122)
plt.imshow(StentContraste)
plt.set_cmap('gray')
plt.title('Stent Contraste')
plt.show()


#%% 9 - Ajuste de contraste com gamma
StentContraste = misc.imAdjust(StentBrilho,0.2,0.7,gamma = 0.5)
hist  = info.doHistogram(StentContraste)
plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('Ajuste de contraste com gamma')
plt.subplot(121)
plt.stem(hist[1,:])

plt.subplot(122)
plt.imshow(StentContraste)
plt.set_cmap('gray')
plt.title('Stent Contraste')
plt.show()


