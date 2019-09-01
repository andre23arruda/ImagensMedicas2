#%% Importando módulos
import numpy as np
import imageio
import matplotlib.pyplot as plt
from skimage import exposure
import os

#%% 1- Lendo imagens
cf = os.getcwd() # current folder
mamo_path = os.path.join(cf,'ImagensAulas','Mamography.pgm') # endereco da imagem
stent_path = os.path.join(cf,'ImagensAulas','Stent.pgm') # endereco da imagem

Mamo = imageio.imread(mamo_path)
Stent = imageio.imread(stent_path)

#%% 2 - Exibindo Mamo
plt.figure()
plt.imshow(Mamo)
plt.set_cmap('gray')
plt.title('Mamo')
plt.show()

#%% 3 - 'for' passando por cada pixel
M,N = Mamo.shape
Mamo_test = Mamo.copy() # criando uma cópia para ficar zero bala
for i in Mamo_test: # Pegando a linha toda de mamo
    for j in i: # pegando cada elemento da linha atual
        Mamo_test[j] = 0 # colocando zero na posição do elemento atual

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
def doHistogram(image):
    maximum = 256
    arr = np.zeros((1,maximum))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            arr[0,image[i,j]] += 1
    return arr

h = doHistogram(Stent)
plt.figure()
plt.stem(np.arange(256),h[0])
plt.get_current_fig_manager().window.showMaximized() # para deixar a figura maximizada
plt.show()

#%% 6' - Outro jeito de fazer histograma
hist2,counts  = np.histogram(Stent,bins = 256)
counts = np.unique(np.round(counts))[np.newaxis,:]
plt.figure()
plt.stem(counts[0],hist2)
plt.show()

#%% 7 - Aumentando brilho
StentBrilho = Stent + 50
hist,counts  = np.histogram(StentBrilho,bins = 256)
counts = np.unique(np.round(counts))[np.newaxis,:]
plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('My title')
plt.subplot(121)
plt.stem(hist)
plt.subplot(122)
plt.imshow(StentBrilho)
plt.set_cmap('gray')
plt.show()

#%% 8 - Ajuste de contraste
StentContraste = exposure.rescale_intensity(StentBrilho, in_range=(0.2*Stent.max(), 0.7*Stent.max()))
hist,counts  = np.histogram(StentContraste,bins = 256)
counts = np.unique(np.round(counts))[np.newaxis,:]
plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('My title')
plt.subplot(121)
plt.stem(hist)
plt.subplot(122)
plt.imshow(StentContraste)
plt.set_cmap('gray')
plt.show()

#%% 9 - Ajuste de contraste com gamma
gamma_corrected = exposure.adjust_gamma(StentContraste, 0.5)
hist,counts  = np.histogram(gamma_corrected,bins = 256)
counts = np.unique(np.round(counts))[np.newaxis,:]
plt.figure()
fig = plt.gcf()
fig.canvas.set_window_title('My title')
plt.subplot(121)
plt.stem(hist)
plt.subplot(122)
plt.imshow(StentContraste)
plt.set_cmap('gray')
plt.show()




