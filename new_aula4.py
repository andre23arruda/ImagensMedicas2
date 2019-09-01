#%% Importando módulos
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
from medImUtils import info, misc,filters,changeFormat
#%% 1 - Ordenando valores
amostra = np.array([15,29, 5, 8, 255, 40, 1, 0, 10])
amostraOrdenada = np.sort(amostra)
mediana = amostraOrdenada[int(len(amostraOrdenada)/2)]
mediana_2 = np.median(amostraOrdenada).astype(int)

#%% 2 - Carregando e filtrando imagem com mediana
imri = imageio.imread(r'ImagensAulas\TransversalMRI_salt-and-pepper.pgm')
IMRIfiltrada = np.zeros(imri.shape,dtype = int)
for indexRow,row in enumerate(imri[:-2]):
    for indexColumn,column in enumerate(row[:-2]):
        windowMedian = imri[indexRow:indexRow+3,indexColumn:indexColumn+3]
        windowMedian = windowMedian.reshape(1,-1)
        windowMedian = np.sort(windowMedian)
        IMRIfiltrada[indexRow+1,indexColumn+1] = windowMedian[0,int((windowMedian.shape[1])/2)]

#%% 2 - Exibindo imagem
images = {'imri':imri,'IMRIfiltrada':IMRIfiltrada}
info.showImageStyle(1,2,images,['Original','IMRIfiltrada'],title='TransversalMRI_salt-and-pepper')
     
#%% 2 - Histogramas     

h1 = info.doHistogram(imri,show = False)
h2 = info.doHistogram(IMRIfiltrada,show = False)
    
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Histograms')
plt.subplot(121)
plt.stem(h1[1,:])
plt.title('Original')
plt.subplot(122)
plt.stem(h2[1,:])
plt.title('Median Filter')
plt.show()   

#%% Usando filtro mediana do modulo scipy
IMRIfiltrada2 = ndimage.median_filter(imri,(3,3))
images = {'IMRIfiltrada':IMRIfiltrada,'IMRIfiltrada2':IMRIfiltrada2}
info.showImageStyle(1,2,images,['IMRIfiltrada','IMRIfiltrada2'],title='Filter Figure')   
# Por que usar vmin e vmax no imshow?
# Esses parâmetros estabelecem a resolução de intensidade
# faixa de valores de intensidade que cada pixel pode representar.
# vmin = 0 a vmax = 255 é o nosso famoso 8bits 

#%%  3 - Função gaussiana
x = 10
u = 5
s = 1

g = misc.gaussmf(x,u,s)

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Gaussian')
plt.subplot(121)
plt.plot(g)
plt.title('Gaussian Distribution')
plt.show()    
g = g[:,np.newaxis] # criando nova dimensão para g
# antes g era (10,)
# agora virou (10,1)   
w_Gauss2D = np.dot(g,np.transpose(g)) # np.dot dá certo para criarmos nossa matriz
w_Gauss2D1 = np.uint8(255*w_Gauss2D)

plt.subplot(122)
plt.imshow(w_Gauss2D1)
plt.title('Gaussian Filter')
plt.show()       

#%% 3 - Convolução mamo e kernel
mamo = imageio.imread(r'ImagensAulas\Mamography.pgm')
w_Gauss2Dnorm = w_Gauss2D/w_Gauss2D.sum()
mamoConv = ndimage.convolve(mamo,w_Gauss2Dnorm)
info.showImageStyle(1,1,{'mamoConv':mamoConv},['Mamo Gaussian Filter'],title='Mamo Gaussian Filter')   

#%% Função mascara gaussiana
w_Gauss2D = filters.kernelGauss(10,5,A=1)
info.showImageStyle(1,1,{'w_Gauss2D':changeFormat.im2uint8(w_Gauss2D)},['w_Gauss2D'])   

#%% Convolucao com mascara gaussiana gerada no item acima
mamoConv = ndimage.convolve(mamo,w_Gauss2D)
info.showImageStyle(1,1,{'mamoConv':mamoConv},['mamoConv'])   

#%% 4 -Afiamento de bordas
imri = imageio.imread(r'ImagensAulas\TransversalMRI2.pgm')
imri = changeFormat.uint2double(imri) # Conversão para double para não dar ruim
mask = filters.kernelGauss(8,3,A=1)
f_blur = ndimage.convolve(imri,mask)
g = imri - f_blur
f_sharpened = (imri + g).clip(0,1)
images = {'imri':changeFormat.im2uint8(imri),'f_blur':changeFormat.im2uint8(f_blur),'f_sharpened':changeFormat.im2uint8(f_sharpened)}
info.showImageStyle(1,3,images,['imri','f_blur','f_sharpened'])   


#%% 5 - Gradientes x e y
dx = np.asarray([[1,-1]])
dy = np.asarray([[1],[-1]])
stent = imageio.imread(r'ImagensAulas\Stent.pgm')
stent = changeFormat.uint2double(stent) # conversão para double para não dar ruim
stent_dx = ndimage.correlate(stent,dx) # derivada em x
stent_dy = ndimage.correlate(stent,dy) # derivada em y
images = {'stent':changeFormat.im2uint8(stent),'stent_dx':changeFormat.im2uint8(stent_dx),'stent_dy':changeFormat.im2uint8(stent_dy)}
info.showImageStyle(1,3,images,['stent','stent_dx','stent_dy'])   

#%% 5 - Priwitt
wx = np.asarray([[-1,0,1],[-1,0,1],[-1,0,1]])
wy = np.asarray([[-1,-1,-1],[0,0,0],[1,1,1]])
stent = imageio.imread(r'ImagensAulas\Stent.pgm')
stent = changeFormat.uint2double(stent) # conversão para double para não dar ruim
stent_x = ndimage.convolve(stent,wx)
stent_y = ndimage.convolve(stent,wy)
gradient_priwitt = np.sqrt(stent_x**2 + stent_y**2)
images = {'stent':changeFormat.im2uint8(stent),'stent_x':changeFormat.im2uint8(stent_x),'stent_y':changeFormat.im2uint8(stent_y),'gradient_priwitt':changeFormat.im2uint8(gradient_priwitt)}
info.showImageStyle(1,4,images,['stent','stent_x','stent_y','gradient_priwitt'])   


#%% 5 - Sobel
wx = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
wy = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
stent = imageio.imread(r'ImagensAulas\Stent.pgm')
stent = (stent - stent.min())/(stent.max() - stent.min()) # conversão para double para não dar ruim
stent_x = ndimage.correlate(stent,wx)
stent_y = ndimage.correlate(stent,wy)
gradient_sobel = np.sqrt(stent_x**2 + stent_y**2)
images = {'stent':changeFormat.im2uint8(stent),'stent_x':changeFormat.im2uint8(stent_x),'stent_y':changeFormat.im2uint8(stent_y),'gradient_sobel':changeFormat.im2uint8(gradient_sobel)}
info.showImageStyle(1,4,images,['stent','stent_x','stent_y','gradient_sobel'])   



#%% 6 - Laplaciano 
L1 = np.asarray([[0,1,0],[1,-4,1],[0,1,0]])
L2 = np.asarray([[1,1,1],[1,-8,1],[1,1,1]])
stent = imageio.imread(r'ImagensAulas\Stent.pgm')
stent = (stent - stent.min())/(stent.max() - stent.min()) # aqui é a conversão para double para não dar ruim
stent_L1 = ndimage.correlate(stent,L1) # Laplaciano 1
stent_L2 = ndimage.correlate(stent,L2) # Laplaciano 2
images = {'stent':changeFormat.im2uint8(stent),'stent_L1':changeFormat.im2uint8(stent_L1),'stent_L2':changeFormat.im2uint8(stent_L2)}
info.showImageStyle(1,3,images,['stent','stent_L1','stent_L2'])   




