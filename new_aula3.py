#%% Importando módulos
import numpy as np
from scipy import ndimage
import imageio
from medImUtils import info

#%% 1 - Dois vetores
w = np.array([1,2,3,2,8])
f = np.array([0,0,0,1,0,0,0,0])
fpadding = np.zeros(len(w)-1,dtype = 'int32')
fpadding = np.hstack((fpadding,f,fpadding)) # concatenando horizontalmente os vetores

#%% 2 - Varredura fpadding
for index,fp in enumerate(fpadding): # enumerate para pegar indices e valores
    print(f'{fp}', end = ' ')


#%% 3 - Correlação
c = np.array([])
for index,fp in enumerate(fpadding[:-4]):
    c = np.hstack((c,np.sum(fpadding[index:index+5] * w)))
    
#%% 4 - Recortando para ficar com o tamanho original
c2 = c[:-4].astype(int)

#%% 5 - Criando matrizes
f,c = np.zeros((5,5),dtype=int),np.zeros((5,5),dtype=int)
f[2,2] = 1
w = np.arange(1,10).reshape(3,-1)

#%% 6 - Correlação matriz
for indexrow,frow in enumerate(f[:-2]): # enumerate para pegar indices e valores
    for indexcolumn,fcolumn in enumerate(frow[:-2]): # enumerate para pegar indices e valores
        corrValue = np.sum(f[indexrow:indexrow+3,indexcolumn:indexcolumn+3] * w)
        c[indexrow+1,indexcolumn+1] = corrValue

#%% 8 - Correlação com ndimage scipy
corr = ndimage.correlate(f,w) # Deu bom

#%% 11 - Correlação 3x3 mamografia
I1 = imageio.imread(r'ImagensAulas\Mamography.pgm')
w = np.ones((3,3)) * 1/9
C_3x3  = ndimage.correlate(I1,w) # Deu bom
images = {'I1':I1,'C_3x3':C_3x3}
info.showImageStyle(1,2,images,['Original','Corr 3x3'],title='Mamo Figure')


#%% 11 - Correlação 5x5 mamografia
I1 = imageio.imread(r'ImagensAulas\Mamography.pgm')
w = np.ones((5,5)) * 1/25
C_5x5  = ndimage.correlate(I1,w) # Deu bom
images = {'I1':I1,'C_5x5':C_5x5}
info.showImageStyle(1,2,images,['Original','Corr 5x5'],title='Mamo Figure')


#%% 11 - Correlação 10x10 mamografia
I1 = imageio.imread(r'ImagensAulas\Mamography.pgm')
w = np.ones((10,10)) * 1/100
C_10x10  = ndimage.correlate(I1,w) # Deu bom
images = {'I1':I1,'C_10x10':C_10x10}
info.showImageStyle(1,2,images,['Original','Corr 10x10'],title='Mamo Figure')












