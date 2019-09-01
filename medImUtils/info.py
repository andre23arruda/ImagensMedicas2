"""
Módulo com funções de informação úteis para a UC de Imagens Biomédicas
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
            plt.figure() # nova figura
            plt.get_current_fig_manager().window.showMaximized() # maximizando
            fig = plt.gcf() # current figure
            fig.canvas.set_window_title('Histogram') # titulo da janela
            plt.title('Histogram') # titulo do grafico
            plt.stem(np.arange(256),h) # plot do tipo stem
            plt.show() # exibindo
        return np.vstack((np.arange(256),h))
    except:
        print('ERRO!Não foi possível gerar o histograma')
        print('Veja se a entrada está adequada')
        
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

