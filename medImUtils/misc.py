"""
Módulo com funções úteis para a UC de Imagens Biomédicas
Autor: André Luiz Costa de Arruda
UNIVERSIDADE FEDERAL DE SÃO PAULO
"""

# -------------------------------------------------------------- #

def imResize(image,p1 = 1,p2 = 1):
    """
    Alterando o tamano da imagem de acordo com a proporcao de entrada
        
    INPUTS:
        image: Imagem que mudará de tamanho
        
        p1: Proporção da largura
        
        p2: Proporção da altura
    """
    from skimage.transform import resize as skResize
    image_resized = skResize(image, (int(image.shape[0] * p1), int(image.shape[1] *p2)), anti_aliasing=True,preserve_range=True)
    return image_resized

    
# -------------------------------------------------------------- #
    
def imBrightness(image,p):
    """
    Altera o brilho da imagem, aumenta ou diminui a intensidade dos pixels
        
    INPUTS:
        image: Imagem a ser normalizada;
        
        p: parametro de aumento de intensidade (entre -255 e 255)
    OUTPUTS:
        imageB: Imagem com o brilho modificado
    """
    import numpy as np
    image = np.float16(image)
    imageB = np.uint8(np.clip(image+p,0,255))
    return imageB

# -------------------------------------------------------------- #
def imAdjust(image,p1,p2,gamma = 1):
    """
    Ajuste de contraste da imagem
        
    INPUTS:
        image: Imagem a ser ajustada;
        
        p1: parametro mínimo de ajuste do contraste
        
        p2: parametro máximo de ajuste do contraste
        
        gamma: parâmetro para mudar a escala do contraste (Default = 1)
    OUTPUTS:
        imageB: Imagem ajustada
    """
    from skimage import exposure
    imageResult = exposure.rescale_intensity(image, in_range=(p1*image.max(), p2*image.max()))
    imageResult = exposure.adjust_gamma(imageResult, gamma)        
    return imageResult

# -------------------------------------------------------------- #
def gaussmf(x,u,s,A=1):
    """
    Criar distribuição gaussiana de acordo com média e desvio padrão
        
    INPUTS:
        x: tamanho da sequencia 
        
        u: média da distribuição gaussiana
        
        s: desvio padrão da distribuição gaussiana
        
        A: amplitude da distribuição (Default: 1)
    OUTPUTS:
        g: distribuição gaussiana com o tamanho de x
    """
    import numpy as np
    x = np.arange(x)
    g = A*np.exp((-0.5*(x-u)**2)/(s**2))
    return g

# -------------------------------------------------------------- #
def tfunc(start = 0, end = 10, step = 0.01, f=1):
    """
    Criar sinal senoidal de acordo com inicio, fim e passo
        
    INPUTS:
        f: frequencia
        
        start: valor de inicio (Default = 0)
        
        end: valor de termino
        
        step: passo da sequencia criada (Default = 0.01)
    OUTPUTS:
        signal: sinal senoidal criado
    """
    import numpy as np
    t = np.arange(start,end+step,step)
    signal = np.sin(2*np.pi*f*t)
    return signal