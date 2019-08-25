"""
Módulo com funções úteis para a UC de Imagens Biomédicas
Autor: André Luiz Costa de Arruda
UNIVERSIDADE FEDERAL DE SÃO PAULO
"""

# -------------------------------------------------------------- #

def shapePrint(imagem):
    """
    Imprime o tamanho da imagem
        
    INPUTS:
        image: Imagem analisada;
    """
    print(f'O tamanho da imagem é: {imagem.shape}')
    
# -------------------------------------------------------------- #
    
def im2double(image):
    """
    Normaliza a imagem para float entre 0 e 1
        
    INPUTS:
        image: Imagem a ser normalizada;
        
    OUTPUTS:
        imageDouble: Imagem normalizada com float entre entre 0 e 1;
    """
    imageDouble = (image-image.min())/(image.max()-image.min())
    return imageDouble

# -------------------------------------------------------------- #
    
def uint2double(image):
    """
    Normaliza a imagem para float entre 0 e 1
        
    INPUTS:
        image: Imagem a ser normalizada;
        
    OUTPUTS:
        imageDouble: Imagem normalizada com float entre entre 0 e 1;
    """
    if image.dtype == 'uint8':
        imageDouble = image/255
    elif image.dtype == 'uint16':
        imageDouble = image/(2**16 - 1)
    return imageDouble

# -------------------------------------------------------------- #

def im2uint8(image):
    """
    Estabelece resolução de intensidade para 8 bits
        
    INPUTS:
        image: Imagem a ser normalizada;
        
    OUTPUTS:
        image8: Imagem com valores de 0 a 255;
    """
    import numpy as np
    image = im2double(image)
    image8 = np.uint8(image*255)
    return image8

# -------------------------------------------------------------- #

def doHistogram(image,show = False):
    """
    Histograma da imagem
        
    INPUTS:
        image: Imagem uint8 a qual vai ser gerado o histograma;
        
    OUTPUTS:
        h: Histograma da imagem com valores de 0 a 255;
    """
    import numpy as np
    import matplotlib.pyplot as plt
    h = np.zeros(256)
    index,counts = np.unique(image,return_counts = True)
    try:
        h[index] = counts
        if show:
            plt.figure()
            plt.get_current_fig_manager().window.showMaximized()
            fig = plt.gcf()
            fig.canvas.set_window_title('Histogram')
            plt.title('Histogram')
            plt.stem(np.arange(256),h)
            plt.show()
        return np.vstack((np.arange(256),h))
    except:
        print('ERRO!Não foi possível gerar o histograma')
        print('Veja se a entrada está adequada')
       
# -------------------------------------------------------------- #

def kernelGauss(u,s,A=1):
    """
    Criar máscara gaussiana
    
    INPUTS:
        u: media;
        
        s: desvio padrão;
        
        A: Amplitude, padrão = 1
    OUTPUTS:
        mascara 2D gaussiana normalizada 
    """
    import numpy as np
    x = np.arange(1,u*2)
    g = A*np.exp((-0.5*(x-u)**2)/(s**2))
    g = g[:,np.newaxis]       
    w_Gauss2D = np.dot(g,np.transpose(g)) 
    return w_Gauss2D/w_Gauss2D.sum()

# -------------------------------------------------------------- #

def showImageStyle(nrows,ncols,images,titleImages,title='Figure'):
    """
    Função de plot padrão de André
    
    INPUTS:
        nrows: número de linhas do subplot;
        
        ncols: número de colunas do subplot
        
        images: Dicionário com a imagens que serão plotadas. Ex: d = {'image1':image1,'image2':image2}
        
        titleImages: lista com o título das imagens
        
        title: Título da janela da figura
    OUTPUTS:
        Figura com subplots de acordo com a entrada
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.get_current_fig_manager().window.showMaximized()
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    count = 0
    for key,value in images.items():
        number = 100 + (ncols*10) + count + 1
        plt.subplot(number)
        plt.imshow(images[key],vmin=0,vmax=255,cmap='gray')
        plt.title(titleImages[count])
        plt.axis('on')
        count += 1
    plt.show()  
    
# -------------------------------------------------------------- #
   
def ffilter(image,f):
    """
    Criando filtro ideal para dominio da frequencia
    
    INPUTS:
        image: imagem original ou imagem no dominio da frequencia
        
    OUTPUTS:
        H: filtro de frequencia
    """
    import numpy as np
    H = np.zeros(image.shape, dtype = int)
    centerX, centerY = int((image.shape[0])/2),int((image.shape[1])/2)
    filter_length = centerX*0.5
    for i in range (H.shape[0]):
        for j in range (H.shape[1]):
            if ((centerX - i)**2 + (centerY - j)**2)**0.5 <= filter_length:
                H[i,j] = 1   
    return H

# -------------------------------------------------------------- #

def imageFFT(image, show = False):
    """
    FFT de imagem
    
    INPUTS:
        image: imagem original para ser passada para o dominio da frequencia
        
        show: parametro para exibição da imagem no domínio da frequencia (padrão: False)
    OUTPUTS:
        fftimage: imagem no dominio da frequencia
        
        fftshift: shift do dominio da frequencia
    """
    import numpy as np
    from medImUtils import imUtils 
    fftimage = np.fft.fft2(image)
    fftshift = np.fft.fftshift(fftimage)
    absfft = imUtils.im2uint8(np.abs(fftshift))
    if show:
        images = {'absfft':absfft}
        imUtils.showImageStyle(1,1,images,['absfft'])
    return fftimage,fftshift,absfft

# -------------------------------------------------------------- #

def imageIFFT(imageFFT, show = False):
    """
    IFFT de imagem
    
    INPUTS:
        imageFFT: imagem no domínio da frequencia
        
        show: parametro para exibição da imagem no domínio espacial (padrão: False)
    OUTPUTS:
        Ifft: imagem no dominio espacial
        
        IfftABS: valor absoluto da imagem no dominio espacial
    """
    import numpy as np
    from medImUtils import imUtils 
    Ifft = np.fft.ifftshift(imageFFT)
    Ifft = np.fft.ifft2(Ifft)
    IfftABS = imUtils.im2uint8(np.abs(Ifft))
    if show:
        images = {'IfftABS':IfftABS}
        imUtils.showImageStyle(1,1,images,['IfftABS'])
    return Ifft,IfftABS,IfftABS

# -------------------------------------------------------------- #

def gaussianFilter2D(m,n,fc,filterType = 'LP',show = False):
    """
    Filtro gaussiano para dominio da frequencia
    
    INPUTS:
        m: número de linhas
        
        n: número de colunas
        
        fc: frequencia de corte (0 a 1)
        
        filterType: Passa baixa ou passa alta (padrão = LP passa baixa)
        
        show: parametro para exibir filtro criado
    OUTPUTS:
        H: filtro gaussiano 2D
    """
    import numpy as np
    from medImUtils import imUtils 
    if filterType == 'LP':
        H = np.zeros((m,n))
    else:
        H = np.full((m,n),-1).astype(float)
    centerX, centerY = int(m/2),int(n/2)
    Do = fc*(0.5*(centerX*0.5 + centerY*0.5))
    for i in range (m):
        for j in range (n):
            D_uv = ((centerX - i)**2 + (centerY - j)**2)**0.5
            H[i,j] = np.abs(H[i,j] + (np.exp((-(D_uv)**2)/(2*Do**2))))
    if show:
        images = {'H':imUtils.im2uint8(H)}
        imUtils.showImageStyle(1,1,images,['Gaussian Mask'])
    return H
            
# -------------------------------------------------------------- #
            
def butterFilter2D(m,n,fc,nPoles,filterType = 'LP',show = False):
    """
    Filtro butterworth para dominio da frequencia
    
    INPUTS:
        m: número de linhas
        
        n: número de colunas
        
        fc: frequencia de corte (0 a 1)
        
        nPoles: número de polos para filtragem
        
        filterType: Passa baixa ou passa alta (padrão = LP passa baixa)
        
        show: parametro para exibir filtro criado
    OUTPUTS:
        H: filtro butterworth 2D
    """
    import numpy as np
    from medImUtils import imUtils 
    if filterType == 'LP':
        H = np.zeros((m,n))
    else:
        H = np.full((m,n),-1).astype(float)
    centerX, centerY = int(m/2),int(n/2)
    Do = fc*(0.5*(centerX*0.5 + centerY*0.5))
    for i in range (m):
        for j in range (n):
            D_uv = ((centerX - i)**2 + (centerY - j)**2)**0.5
            H[i,j] = np.abs(H[i,j] + (1/(1+(D_uv/Do)**(2*nPoles))))
    if show:
        images = {'H':imUtils.im2uint8(H)}
        imUtils.showImageStyle(1,1,images,['Butter Mask'])
    return H            

# -------------------------------------------------------------- #
            
def idealfilter(m,n,fc, filterType = 'LP',show = False):
    """
    Filtro ideal para dominio da frequencia
    
    INPUTS:
        m: número de linhas
        
        n: número de colunas
        
        fc: frequencia de corte (0 a 1)
                
        filterType: Passa baixa ou passa alta (padrão = LP passa baixa)
        
        show: parametro para exibir filtro criado
    OUTPUTS:
        H: filtro ideal 2D
    """
    import numpy as np
    from medImUtils import imUtils 
    H = np.zeros((m,n), dtype = int)
    centerX, centerY = int(m/2),int(n/2)
    Do = fc*(0.5*(centerX*0.5 + centerY*0.5))
    for i in range (H.shape[0]):
        for j in range (H.shape[1]):
            D_uv = ((centerX - i)**2 + (centerY - j)**2)**0.5
            if D_uv <= Do and filterType == 'LP':
                H[i,j] = 1  
            elif D_uv >= Do and filterType != 'LP':
                H[i,j] = 1
    if show:
        images = {'H':imUtils.im2uint8(H)}
        imUtils.showImageStyle(1,1,images,['Butter Mask'])
    return H

# -------------------------------------------------------------- #
    
def priwitt(image):
    """
    Filtro Priwitt
    
    INPUTS:
        image: imagem a ser filtrada
    OUTPUTS:
        gradient_priwitt: gradiente de Priwitt resultante das derivadas x e y
    """
    import numpy as np
    from medImUtils import imUtils
    from scipy import ndimage
    wx = np.asarray([[-1,0,1],[-1,0,1],[-1,0,1]])
    wy = np.asarray([[-1,-1,-1],[0,0,0],[1,1,1]])
    image_norm = imUtils.im2double(image)
    dx = ndimage.convolve(image_norm,wx)
    dy = ndimage.convolve(image_norm,wy)
    gradient_priwitt = np.sqrt(dx**2 + dy**2)
    return gradient_priwitt

# -------------------------------------------------------------- #
    
def sobel(image):
    """
    Filtro Sobel
    
    INPUTS:
        image: imagem a ser filtrada
    OUTPUTS:
        gradient_sobel: gradiente de Sobel resultante das derivadas x e y
    """
    import numpy as np
    from medImUtils import imUtils
    from scipy import ndimage
    wx = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    wy = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
    image_norm = imUtils.im2double(image)
    dx = ndimage.convolve(image_norm,wx)
    dy = ndimage.convolve(image_norm,wy)
    gradient_sobel = np.sqrt(dx**2 + dy**2)
    return gradient_sobel
    
    
# -------------------------------------------------------------- #
    
def laplace(image):
    """
    Filtros Laplacianos
    
    INPUTS:
        image: imagem a ser filtrada
    OUTPUTS:
        dL1: Laplaciano resultante da máscara [[0,1,0],[1,-4,1],[0,1,0]]
        
        dL2: Laplaciano resultante da máscara [[1,1,1],[1,-8,1],[1,1,1]]
    """
    import numpy as np
    from medImUtils import imUtils
    from scipy import ndimage
    L1 = np.asarray([[0,1,0],[1,-4,1],[0,1,0]])
    L2 = np.asarray([[1,1,1],[1,-8,1],[1,1,1]])
    image_norm = imUtils.im2double(image)
    dL1 = ndimage.convolve(image_norm,L1)
    dL2 = ndimage.convolve(image_norm,L2)
    return dL1,dL2   

# -------------------------------------------------------------- #
    
def FT(signal,time,f):
    """
    Transformada de Fourier bostinha de um sinal
    
    INPUTS:
        signal: sinal a ser passado para o dominio da frequencia
        
        time: tempo do sinal adquirido
        
        f: limite de fequencias que devem ser calculadas
    OUTPUTS:
        X: sinal no dominio da frequencia
    """
    import numpy as np
    X = np.zeros(f+1,dtype = complex)
    for i in range(f+1):
        X[i] = np.sum(signal*np.exp((-1j)*2*np.pi*i*time))
    return X    

# -------------------------------------------------------------- #

def meanFilter(image,kernelLength):
    """
    Filtro média
    
    INPUTS:
        image: imagem a ser filtrada
        
        kernelLength: tamanho da máscara de filtragem, deve ser ímpar
        
    OUTPUTS:
        newImage: imagem filtrada
    """
    import numpy as np
    newImage = np.zeros(image.shape)
    w = np.zeros((kernelLength,kernelLength),dtype=int)
    w_center = int((w.shape[0])/2)
    for indexrow,frow in enumerate(image[:-(w_center*2)]):
        for indexcolumn,fcolumn in enumerate(frow[:-(w_center*2)]): 
            meanMask = np.mean(image[indexrow:1+indexrow+w_center*2,indexcolumn:1+indexcolumn+w_center*2] + w)
            newImage[indexrow+w_center,indexcolumn+w_center] = meanMask
    return newImage

# -------------------------------------------------------------- #

def LeeFilter(image,kernelLength,show = False):
    import numpy as np
    import cv2
    from medImUtils import imUtils

    image = imUtils.im2double(image)
    rect = cv2.selectROI(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    c1,c2 = rect[0],rect[0]+rect[2]
    l1,l2 = rect[1],rect[0]+rect[3]
    stdRect = np.std(image[l1:l2,c1:c2])
    newImage = np.zeros(image.shape)
    w = np.zeros((kernelLength,kernelLength),dtype=int)
    w_center = int((w.shape[0])/2)
    for indexrow,frow in enumerate(image[:-(w_center*2)]):
        for indexcolumn,fcolumn in enumerate(frow[:-(w_center*2)]): 
            maskLocal = image[indexrow:1+indexrow+w_center*2,indexcolumn:1+indexcolumn+w_center*2]
            meanMask = np.mean(maskLocal + w)
            k = np.clip(1-(stdRect/(0.001 + maskLocal.std())),0,1)
            newImage[indexrow+w_center,indexcolumn+w_center] = meanMask + k*(image[indexrow+w_center,indexcolumn+w_center] - meanMask)
    if show:
        imUtils.showImageStyle(1,2,{'I1':imUtils.im2uint8(image),'Imean':imUtils.im2uint8(newImage)},['Original','Lee Filter'])
    return newImage

# -------------------------------------------------------------- #

def meanFilterFast(image,kernelLength):
    """
    Filtro média rápido pq é usado convolução do numpy
    
    INPUTS:
        image: imagem a ser filtrada
        
        kernelLength: tamanho da máscara de filtragem, deve ser ímpar
        
    OUTPUTS:
        newImage: imagem filtrada
    """
    import numpy as np
    from scipy import ndimage
    w = np.ones((kernelLength,kernelLength),dtype=int)
    newImage = ndimage.convolve(image,w)
    
    return newImage

# -------------------------------------------------------------- #
def imnoise(image,variance,show = False):
    """
    Adicionar ruído gaussiano na imagem
    
    INPUTS:
        image: imagem na qual será adicionado o ruído
        
        variance: variancia da distribuição gaussiana
        
        show: parâmetro para a imagem com ruído ser exibida (padrão = False)
        
    OUTPUTS:
        Inoisy: imagem com ruído
    """
    import numpy as np
    from medImUtils import imUtils
    noisy = 1*np.random.normal(0,variance**0.5,image.shape)
    Inoisy = imUtils.im2uint8(np.clip((image + noisy),0,1))
    if show:
        imUtils.showImageStyle(1,1,{'Inoisy':Inoisy},['Gaussian Noise Var: '+str(variance)])
    return Inoisy

