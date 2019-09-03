# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:54:52 2019

@author: Andre
"""

#%% Importando módulos
import imageio
from medImUtils import filters,info,changeFormat
import os

#%% 1 - Funções de transferência - Filtro ideal
m,n,fc,ft1,show = 400,400,0.2,'LP',True
H_ideal = filters.idealfilter(m,n,fc,ft1,show)

#%% 2 - Funções de transferência - Filtro gaussiano
m,n,fc,ft1,show = 400,400,0.2,'LP',True
H_gaussian = filters.gaussianFilter2D(m,n,fc,ft1,show)

#%% 3 - Funções de transferência - Filtro butterworth
m,n,fc,ft1,show = 400,400,0.2,'LP',True
nPoles = 3
H_butter = filters.butterFilter2D(m,n,fc,nPoles,ft1,show)

#%% 1 - Filtragens Passa baixa
cf = os.getcwd() # current folder
mamo_path = os.path.join(cf,'ImagensAulas','Mamography.pgm') # endereco da imagem
mamo = imageio.imread(mamo_path)

mamoFFT,mamoFFTshift,mamoFFTabs = filters.imageFFT(mamo)
m,n,fc = mamoFFTshift.shape[0],mamoFFTshift.shape[1],0.2

H_ideal = filters.idealfilter(m,n,fc)
mamoFFT_filtered = mamoFFTshift * H_ideal
mamoIFFT,mamoIdeal = filters.imageIFFT(mamoFFT_filtered)

H_gaussian = filters.gaussianFilter2D(m,n,fc)
mamoFFT_filtered = mamoFFTshift * H_gaussian
mamoIFFT,mamoGaussian = filters.imageIFFT(mamoFFT_filtered)

H_butter = filters.butterFilter2D(m,n,fc,2)
mamoFFT_filtered = mamoFFTshift * H_butter
mamoIFFT,mamoButter = filters.imageIFFT(mamoFFT_filtered)

info.showImageStyle(1,4,{'mamo':(mamo),'mamoIdeal':changeFormat.im2uint8(mamoIdeal),
                         'mamoGaussian':changeFormat.im2uint8(mamoGaussian),
                         'mamoButter':changeFormat.im2uint8(mamoButter)},
                    ['mamo','Ideal','Gaussian','Butter'])

#%% 2 - Filtragens Passa alta
mamo = imageio.imread(mamo_path)

mamoFFT,mamoFFTshift,mamoFFTabs = filters.imageFFT(mamo)
m,n,fc = mamoFFTshift.shape[0],mamoFFTshift.shape[1],0.2

H_ideal = filters.idealfilter(m,n,fc,filterType = 'HP')
mamoFFT_filtered = mamoFFTshift * H_ideal
mamoIFFT,mamoIdeal = filters.imageIFFT(mamoFFT_filtered)

H_gaussian = filters.gaussianFilter2D(m,n,fc,filterType = 'HP')
mamoFFT_filtered = mamoFFTshift * H_gaussian
mamoIFFT,mamoGaussian = filters.imageIFFT(mamoFFT_filtered)

H_butter = filters.butterFilter2D(m,n,fc,2,filterType = 'HP')
mamoFFT_filtered = mamoFFTshift * H_butter
mamoIFFT,mamoButter = filters.imageIFFT(mamoFFT_filtered)

info.showImageStyle(1,4,{'mamo':mamo,'mamoIdeal':changeFormat.im2uint8(mamoIdeal),
                         'mamoGaussian':changeFormat.im2uint8(mamoGaussian),
                         'mamoButter':changeFormat.im2uint8(mamoButter)},
                    ['mamo','Ideal','Gaussian','Butter'])


