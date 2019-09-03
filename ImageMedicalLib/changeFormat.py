"""
Módulo com funções úteis de transformar formato de valores para a UC de Imagens Biomédicas
Autor: André Luiz Costa de Arruda
UNIVERSIDADE FEDERAL DE SÃO PAULO
"""

# -------------------------------------------------------------- #
    
def imNormalize(image):
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
    Passa valores inteiros de 0 a 255 para float entre 0 e 1
        
    INPUTS:
        image: Imagem a ser normalizada;
        
    OUTPUTS:
        imageDouble: Imagem normalizada com float entre entre 0 e 1;
    """
    imageDouble = image/255
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
    image = np.clip(image,0,1) # valores entre 0 e 1
    image = imNormalize(image) # menor valor é zero, valor maximo é um
    image8 = np.uint8(image*255) # passando para uint8
    return image8



