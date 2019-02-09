import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def gravar_arquivo_arff(base_teste, classes):
    tam = len(base_teste[0][0])
    file = open('leaf_deseases.arff','w') 
 
    file.write('@relation leaf_deseases\n') 
    for i in range(tam):
        file.write('@attribute '+ str(i) +' NUMERIC\n') 
    
    file.write('@attribute classes {')
    
    a = set(classes)
    
    for i in a:
        file.write(str(i)+',')
    
    file.write('}')    
    
    for i in range(tam):
         len(set(classes))
    
    file.write('\n@data\n') 

    for item in base_teste:
        for i in range(len(item[0])):
            file.write("%s," % str(item[0][i])) 
        file.write("%s\n" % item[1])    
 
    file.close() 
    print('arquivo gravado')

def histograma(img):
    '''
    retorna histograma de cores da imagem
    '''
    WB = np.zeros(256)
    WG = np.zeros(256)
    WR = np.zeros(256)
    
    l, c, tres = img.shape
    
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]
    
    for i in range(l):
        for j in range(c):
            WB[B[i,j]] = WB[B[i,j]]+1
            WG[G[i,j]] = WG[G[i,j]]+1 
            WR[R[i,j]] = WR[R[i,j]]+1 
    for i in range(256):
        WB[i] = WB[i]/(l*c)
        WG[i] = WG[i]/(l*c)
        WR[i] = WR[i]/(l*c)
    return np.append(np.append(WB,WG),WR)

def distancia(a,b):
    '''
    calcula distancia/similaridade entre os histogramas
    '''
    M = len(a)
    soma = 0
    for i in range(M):
        soma = soma + ((a[i]-b[i])**2)
    return np.sqrt(soma)

def lbp(img):
    '''
    aplica convolução ...
    '''
    l, c = img.shape
    img2 = np.zeros((l,c), dtype=int)
    
    for i in range(1,l-1):
        for j in range(1,c-1):
            A = img[i-1,j] #bit mais significativo
            B = img[i-1,j+1]
            C = img[i,j+1]
            D = img[i+1,j+1]
            E = img[i+1,j]
            F = img[i+1,j-1]
            G = img[i,j-1]
            H = img[i-1,j-1]
            Centro = img[i,j]
            soma = 0
            if(A > Centro):
                soma = soma + (2**7)
            if(B > Centro):
                soma = soma + (2**6)
            if(C > Centro):
                soma = soma + (2**5)
            if(D > Centro):
                soma = soma + (2**4)
            if(E > Centro):
                soma = soma + (2**3)
            if(F > Centro):
                soma = soma + (2**2)
            if(G > Centro):
                soma = soma + (2**1)
            if(H > Centro):
                soma = soma + (2**0)
                
            img2[i,j] = soma
    return img2

def histograma_tons_cinza(img):   
    W = np.zeros(256)
    qtdeLinhas, qtdeColunas = img.shape
    
    for i in range(qtdeLinhas):
        for j in range(qtdeColunas):
            W[img[i,j]] = W[img[i,j]] + 1
    
    for i in range(256):
        W[i] = W[i]/(qtdeLinhas*qtdeColunas)
    
    return W

def extrair_caracteristica(img):
    l, c, t = img.shape
    
    qtd_verde = 0
    qtd_azul = 0
    qtd_vermelho = 0

    for i in range(l):
        for j in range(c):
            if(img[i,j,1] > img[i,j,0] and img[i,j,1] > img[i,j,2]):
                qtd_verde += 1
            if(img[i,j,2] > img[i,j,0] and img[i,j,2] > img[i,j,1]):
                qtd_vermelho += 1
            if(img[i,j,0] > img[i,j,1] and img[i,j,0] > img[i,j,2]):
                qtd_azul += 1

    qtd_verde2 = qtd_verde/(qtd_vermelho+qtd_verde+qtd_vermelho)
    qtd_vermelho2 = qtd_vermelho/(qtd_vermelho+qtd_verde+qtd_vermelho)
    qtd_azul2 = qtd_azul/(qtd_vermelho+qtd_verde+qtd_vermelho)
    
    return qtd_verde2+qtd_azul2+qtd_vermelho2

def extrair_caracteristica2(img):
    histB = cv2.calcHist([img],[0],None,[10],[240,250])
    histG = cv2.calcHist([img],[1],None,[10],[240,250])
    histR = cv2.calcHist([img],[2],None,[10],[240,250])
    cv2.normalize(histB, histB)
    cv2.normalize(histG, histG)
    cv2.normalize(histR, histR)
    histFinal = np.append(np.append(histB, histG),histR)
    return histFinal

def get_all_image_paths_by_class():
    BASE_PATH = './grape_base/'
    black_measles_imgs = os.listdir(BASE_PATH+'black_measles')
    black_measles_imgs = [BASE_PATH+'black_measles/'+img for img in black_measles_imgs]

    leaf_blight_imgs = os.listdir(BASE_PATH+'leaf_blight')
    leaf_blight_imgs = [BASE_PATH+'leaf_blight/'+img for img in leaf_blight_imgs]

    black_rot_imgs = os.listdir(BASE_PATH+'black_rot')
    black_rot_imgs = [BASE_PATH+'black_rot/'+img for img in black_rot_imgs]

    healthy_imgs = os.listdir(BASE_PATH+'healthy')
    healthy_imgs = [BASE_PATH+'healthy/'+img for img in healthy_imgs]
    
    classes = {}
    classes['black_measles'] = black_measles_imgs
    classes['leaf_blight'] = leaf_blight_imgs
    classes['black_rot'] = black_rot_imgs
    classes['healthy'] = healthy_imgs
    
    return classes