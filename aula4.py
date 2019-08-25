#%% Importando módulos
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
import os

#%% 1 - Ordenando valores
amostra = np.array([15,29, 5, 8, 255, 40, 1, 0, 10])
amostraOrdenada = np.sort(amostra)
mediana = amostraOrdenada[int(len(amostraOrdenada)/2)]
mediana_2 = np.median(amostraOrdenada).astype(int)

#%% 2 - Carregando e filtrando imagem
cf = os.getcwd() # current folder
imri_path = os.path.join(cf,'ImagensAulas','TransversalMRI_salt-and-pepper.pgm') # endereco da imagem
imri = imageio.imread(imri_path)
IMRIfiltrada = np.zeros(imri.shape,dtype = int)

for indexRow,row in enumerate(imri[:-2]):
    for indexColumn,column in enumerate(row[:-2]):
        windowMedian = imri[indexRow:indexRow+3,indexColumn:indexColumn+3]
        windowMedian = windowMedian.reshape(1,-1)
        windowMedian = np.sort(windowMedian)
        IMRIfiltrada[indexRow+1,indexColumn+1] = windowMedian[0,int((windowMedian.shape[1])/2)]

#%% 2 - Exibindo imagem
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Filter Figure')
plt.subplot(121)
plt.imshow(imri)
plt.title('Original')
plt.set_cmap('gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(IMRIfiltrada)
plt.title('Median Filter')
plt.set_cmap('gray')
plt.axis('off')
plt.show()        
        
#%% Histogramas     
h,hFilt = np.zeros((256)),np.zeros((256))
values,counts = np.unique(imri,return_counts=True) 
h[values] = counts       
values,counts = np.unique(IMRIfiltrada,return_counts=True) 
hFilt[values] = counts    

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Histograms')
plt.subplot(121)
plt.stem(h)
plt.title('Original')
plt.subplot(122)
plt.stem(hFilt)
plt.title('Median Filter')
plt.show()   

#%% Usando filtro mediana do modulo scipy
IMRIfiltrada2 = ndimage.median_filter(imri,(3,3))

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Filter Figure')
plt.subplot(121)
plt.imshow(IMRIfiltrada,vmin=0,vmax=255,cmap='gray')
plt.title('Median Filter Manual')
plt.axis('off')
plt.subplot(122)
plt.imshow(IMRIfiltrada2,vmin=0,vmax=255,cmap='gray')
plt.title('Median Filter Scipy')
plt.axis('off')
plt.show()       
# Por que usar vmin e vmax no imshow?
# Esses parâmetros estabelecem a resolução de intensidade
# faixa de valores de intensidade que cada pixel pode representar.
# vmin = 0 a vmax = 255 é o nosso famoso 8bits 

#%%  3 - Função gaussiana
x = np.arange(1,10,1)
u = 5
s = 1

def gaussmf(x,u,s,A=1):
    g = A*np.exp((-0.5*(x-u)**2)/(s**2))
    return g

g = gaussmf(x,u,s,A=1)
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Gaussian')
plt.plot(x,g)
plt.title('Median Filter Manual')
plt.show()    
g = g[:,np.newaxis]       
w_Gauss2D = np.dot(g,np.transpose(g)) 
w_Gauss2D1 = np.uint8(255*w_Gauss2D)

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Gaussian Filter')
plt.imshow(w_Gauss2D1)
plt.title('Gaussian Filter')
plt.show()       

#%% Convolução mamo e kernel
cf = os.getcwd() # current folder
mamo_path = os.path.join(cf,'ImagensAulas','Mamography.pgm') # endereco da imagem
mamo = imageio.imread(mamo_path)
w_Gauss2Dnorm = w_Gauss2D/w_Gauss2D.sum()
mamoConv = ndimage.convolve(mamo,w_Gauss2Dnorm)

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Mamo Gaussian Filter')
plt.imshow(mamoConv)
plt.title('Mamo Gaussian Filter')
plt.show()    

#%% Função mascara gaussiana
def kernelGauss(u,s,A=1):
    x = np.arange(1,u*2)
    g = A*np.exp((-0.5*(x-u)**2)/(s**2))
    g = g[:,np.newaxis]       
    w_Gauss2D = np.dot(g,np.transpose(g)) 
#   w_Gauss2D2 = np.kron(g,np.transpose(g)) # Convolução outro jeito
    return w_Gauss2D/w_Gauss2D.sum()

w_Gauss2D = kernelGauss(10,5,A=1)
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Gaussian Filter')
plt.imshow(w_Gauss2D)
plt.title('Mamo Gaussian Filter')
plt.show()    

#%% Convolucoes
w_Gauss2Dnorm = w_Gauss2D/w_Gauss2D.sum()
mamoConv = ndimage.convolve(mamo,w_Gauss2Dnorm)

plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Mamo Gaussian Filter')
plt.imshow(mamoConv)
plt.title('Mamo Gaussian Filter')
plt.show()

#%% 4 -Afiamento de bordas
imri_path = os.path.join(cf,'ImagensAulas','TransversalMRI2.pgm') # endereco da imagem
imri = imageio.imread(imri_path)
imri = (imri - imri.min())/(imri.max() - imri.min()) # aqui é a conversão para double para não dar ruim
mask = kernelGauss(8,3,A=1)
f_blur = ndimage.convolve(imri,mask)
g = imri - f_blur
f_sharpened = (imri + g).clip(0,1)
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Sharping image')
plt.subplot(131)
plt.imshow(np.uint8(imri*255),vmin = 0,vmax = 255)
plt.title('Original')
plt.subplot(132)
plt.imshow(np.uint8(f_blur*255),vmin = 0,vmax = 255)
plt.title('Gauss Filter 15x15')
plt.subplot(133)
plt.imshow(np.uint8(f_sharpened*255),vmin = 0,vmax = 255)
plt.title('Sharpened')
plt.show()

#%% 5 - Gradientes x e y
dx = np.asarray([[1,-1]])
dy = np.asarray([[1],[-1]])
stent_path = os.path.join(cf,'ImagensAulas','Stent.pgm') # endereco da imagem
stent = imageio.imread(stent_path)
stent = (stent - stent.min())/(stent.max() - stent.min()) # aqui é a conversão para double para não dar ruim
stent_dx = ndimage.correlate(stent,dx) # derivada em x
stent_dy = ndimage.correlate(stent,dy) # derivada em y
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Gradients x and y')
plt.subplot(131)
plt.imshow(np.uint8(stent*255),vmin = 0,vmax = 255)
plt.title('Original')
plt.subplot(132)
plt.imshow(np.uint8((stent_dx*255).clip(0,255)),vmin = 0,vmax = 255)
plt.title('Gradient x')
plt.subplot(133)
plt.imshow(np.uint8((stent_dy*255).clip(0,255)),vmin = 0,vmax = 255)
plt.title('Gradient y')
plt.show()

#%% 5 - Priwitt
wx = np.asarray([[-1,0,1],[-1,0,1],[-1,0,1]])
wy = np.asarray([[-1,-1,-1],[0,0,0],[1,1,1]])
stent = imageio.imread(stent_path)
stent = (stent - stent.min())/(stent.max() - stent.min()) # aqui é a conversão para double para não dar ruim
stent_x = ndimage.convolve(stent,wx)
stent_y = ndimage.convolve(stent,wy)
gradient_priwitt = np.sqrt(stent_x**2 + stent_y**2)
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Priwitt gradient')
plt.subplot(141)
plt.imshow(np.uint8(stent*255),vmin = 0,vmax = 255)
plt.title('Original')
plt.subplot(142)
plt.imshow(np.uint8(stent_x*255),vmin = 0,vmax = 255)
plt.title('Priwitt gradient x')
plt.subplot(143)
plt.imshow(np.uint8(stent_y*255),vmin = 0,vmax = 255)
plt.title('Priwitt gradient y')
plt.subplot(144)
plt.imshow(np.uint8(gradient_priwitt*255),vmin = 0,vmax = 255)
plt.title('Priwitt gradient total')
plt.show()

#%% 5 - Sobel
wx = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
wy = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
stent = imageio.imread(stent_path)
stent = (stent - stent.min())/(stent.max() - stent.min()) # aqui é a conversão para double para não dar ruim
stent_x = ndimage.correlate(stent,wx)
stent_y = ndimage.correlate(stent,wy)
gradient_sobel = np.sqrt(stent_x**2 + stent_y**2)
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Sobel gradient')
plt.subplot(141)
plt.imshow(np.uint8(stent*255),vmin = 0,vmax = 255)
plt.title('Original')
plt.subplot(142)
plt.imshow(np.uint8(stent_x*255),vmin = 0,vmax = 255)
plt.title('Sobel gradient x')
plt.subplot(143)
plt.imshow(np.uint8(stent_y*255),vmin = 0,vmax = 255)
plt.title('Sobel gradient y')
plt.subplot(144)
plt.imshow(np.uint8(gradient_sobel*255),vmin = 0,vmax = 255)
plt.title('Sobel gradient total')
plt.show()

#%% 6 - Laplaciano 
L1 = np.asarray([[0,1,0],[1,-4,1],[0,1,0]])
L2 = np.asarray([[1,1,1],[1,-8,1],[1,1,1]])
stent = imageio.imread(stent_path)
stent = (stent - stent.min())/(stent.max() - stent.min()) # aqui é a conversão para double para não dar ruim
stent_L1 = ndimage.correlate(stent,L1) # Laplaciano 1
stent_L2 = ndimage.correlate(stent,L2) # Laplaciano 2
plt.figure()
plt.get_current_fig_manager().window.showMaximized()
fig = plt.gcf()
fig.canvas.set_window_title('Laplaciano gradient')
plt.subplot(131)
plt.imshow(np.uint8(stent*255),vmin = 0,vmax = 255)
plt.title('Original')
plt.subplot(132)
plt.imshow(np.uint8((stent_L1*255).clip(0,255)),vmin = 0,vmax = 255)
plt.title('Laplaciano gradient 1')
plt.subplot(133)
plt.imshow(np.uint8((stent_L2*255).clip(0,255)),vmin = 0,vmax = 255)
plt.title('Laplaciano gradient 2')
plt.show()




