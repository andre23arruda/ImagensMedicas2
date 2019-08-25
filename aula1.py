#%% Importando módulos necessários
import numpy as np
import matplotlib.pyplot as plt # exibir graficos e imagens
import imageio # Módulo para ler e salvar imagens
import os

#%% 5 - Carregando Imagens
cf = os.getcwd() # current folder
I1_path = os.path.join(cf,'ImagensAulas','raioXTorax.pgm') # endereco da imagem
I1 = imageio.imread(I1_path) # lendo a imagem
I1.shape # tamanho da imagem
I2_path = os.path.join(cf,'ImagensAulas','raioXTorax.jpg')
I2 = imageio.imread(I2_path)
I2.shape

#%% 6 - Função que indica o tamanho da imagem
def shapeFunc(imagem):
    """
        Função que imprime o tamanho da imagem
    """
    print(f'O tamanho da imagem é: {imagem.shape}')
     
shapeFunc(I1)

#%% 7 - Pixel na posicao 50x50
I50x50 = I1[50,50]
print(f'O valor do pixel 50x50 é: {I50x50}')

#%% 8 - Maximo, minimo e media da imagem
maximum = I1.max()
minimum = I1.min()
meanI1 = np.uint8(np.mean(I1))

#%% 9 - Exibindo imagem
plt.figure()
plt.imshow(I1)
plt.set_cmap('gray')
plt.axis('off')
plt.title('Raio X Torax')
plt.colorbar()
plt.show()


#%% 10 - Convertendo para double
I1double = np.double((I1-minimum)/(maximum-minimum))
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
from skimage.transform import resize as skResize # modulo para redimensionar imagens
image_resized1 = skResize(I1, (I1.shape[0] * 2, I1.shape[1] *2), anti_aliasing=True,preserve_range=True)
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