#%% Importando módulos necessários
import numpy as np # Módulo de cálculo matricial do Python
import matplotlib.pyplot as plt # Módulo para exibir graficos e imagens
import imageio # Módulo para ler e salvar imagens
from ImageMedicalLib import info,changeFormat,misc

#%% 5 - Carregando Imagens
I1 = imageio.imread(r'ImagensAulas\raioXTorax.pgm')
I1.shape
I2 = imageio.imread(r'ImagensAulas\raioXTorax.jpg')
I2.shape

#%% 6 - Tamanho da imagem
info.shapePrint(I1)

#%% 7 - Pixel na posicao 50x50
I50x50 = I1[50,50]
print(f'O valor do pixel 50x50 é: {I50x50}')

#%% 8 - Maximo, minimo e media da imagem
maximum = I1.max()
minimum = I1.min()
meanI1 = int(I1.mean())

#%% 9 - Exibindo imagem
plt.figure()
plt.imshow(I1)
plt.set_cmap('gray')
plt.axis('off')
plt.title('Raio X Torax')
plt.colorbar()
plt.show()


#%% 10 - Convertendo para double
I1double = changeFormat.uint2double(I1)
newMax = I1double.max()
newMin = I1double.min()
newMean = I1double.mean()
newStd = I1double.std()
newMedian = np.median(I1double)

plt.figure()
plt.imshow(I1double)
plt.set_cmap('gray')
plt.axis('off')
plt.title('Raio X Torax double')
plt.colorbar()
plt.show()

Idouble50x50 = I1double[50,50]

#%% 11 - Imagem com o tamanho dobrado
image_resized1 = misc.imResize(I1,2,2)
plt.figure()
plt.imshow(image_resized1)
plt.set_cmap('gray')
plt.axis('off')
plt.title('Raio X Torax double')
plt.colorbar()
plt.show()

#%% 12 - Usando a imagem RGB
I2R = I2[:,:,1]
plt.figure()
plt.imshow(I2R)
plt.colorbar()
plt.title('RGB canal R')
plt.set_cmap('gray')
plt.show()