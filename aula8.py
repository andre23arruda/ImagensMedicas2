#%% Importando módulos
import numpy as np
import matplotlib.pyplot as plt
import imageio
from medImUtils import imUtils
import os

#%% 1 - Carregando imagem
cf = os.getcwd() # current folder
I1_path = os.path.join(cf,'ImagensAulas','ImSemRuido.pgm') # endereco da imagem
I1 = imUtils.uint2double(imageio.imread(I1_path))

plt.imshow(I1,cmap='gray')
plt.show()
#%% 1 -Adicionando ruído
Inoise = {}
start = 0.005
for i in range(10):
    variance = np.round(start + (i*start),4)
    noisy = np.random.normal(0,variance**0.5,I1.shape)
    Inoise[str(variance)] = np.clip((I1 + noisy),0,1)
    plt.subplot(2,5,i+1)
    plt.imshow(Inoise[str(variance)])
    plt.title('Variance: ' + str(variance))
    plt.axis('off')
plt.show()






