#Importation des bibliothèques
import numpy as np   
from matplotlib import pyplot as plt  
import cv2
import os.path
import pickle
from scipy.spatial import distance
import time
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path


pathDataset = "static/img"
pathFichierTrain = "train"


def Gris(chemin):
    gray = cv2.imread(chemin,cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return im


#Normalisation de l'image
def normalisationImage(image):
    normImage = image//8
    normImage = normImage.astype('uint32')
    return normImage

#Calcul des moment de HU
def momentHu(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    return huMoments

#Calcul de l'histogramme et normalisation connaissant le chemin de l'image
def Histogramme(chemin):
    image = cv2.imread(chemin)
    histogramme = cv2.calcHist([image], [0, 1, 2], None, [8,8,8],[0, 256, 0, 256, 0, 256])
    histogramme = cv2.normalize(histogramme, histogramme).flatten()
    return histogramme

#Creation du fichier pour stockage des descripteurs pickle
def pickle_hist(fichier,histogramme):   
    pkl=pickle.Pickler(fichier)
    pkl.dump(histogramme)

#Récupération du descripteur Unpickle
def unpickle_hist(fichier):   
    Unpkl=pickle.Unpickler(fichier)
    fic=(Unpkl.load())
    return fic

#Fonction de stockage des histogrammes normalisées
def ApprentissageTexture():
    #Création du fichier de stockage des histogrammes
    f = open((pathFichierTrain+"/histogramme"+".txt"),'wb')
    listeImage = os.listdir(pathDataset)
    hist_obj = {}
    for image in listeImage:
        histogramme = Histogramme(pathDataset+"/"+image)
        hist_obj[image] = histogramme
    pickle_hist(f,hist_obj)
    f.close

def ApprentissageColor():
    #Création du fichier de stockage des histogrammes
    f = open((pathFichierTrain+"/moment"+".txt"),'wb')
    listeImage = os.listdir(pathDataset)
    moment_obj = {}
    for image in listeImage:
        chemin = (pathDataset+"/"+image)
        gris = Gris(chemin)
        moment = momentHu(gris)
        moment_obj[image] = moment
    pickle_hist(f,moment_obj)
    f.close

#Fonction de calcul de distance entre deux histogrammes   
def CalculDistance(h1,h2):
    distances = distance.euclidean(h1,h2)
    return distances

def RessemblaceImageColor(cheminImageTest,k):
    listeDistance ={}
    #Calcul des moments de hu de l'image de test
    gris = Gris(cheminImageTest)
    moment_test = momentHu(gris)
    f = open((pathFichierTrain+"/moment"+".txt"),"rb")
    list_hist = unpickle_hist(f)
    for key, valeur in list_hist.items():
        d = CalculDistance(valeur[0],moment_test[0])
        listeDistance[d] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t:t[0])
    f.close
    
    return(listeDistances[:k])   


#Chercher les plus proches voisins
def RessemblaceImageTexture(cheminImageTest,k):
    listeDistance ={}
    #Calcul de l'histogramme de l'image de test
    h1 = Histogramme(cheminImageTest)
    h1 = np.asanyarray(h1)
    f = open((pathFichierTrain+"/histogramme"+".txt"),"rb")
    list_hist = unpickle_hist(f)
    for key, valeur in list_hist.items():
        valeur = np.asanyarray(valeur)
        CalculDistance(valeur,h1)
        listeDistance[CalculDistance(valeur,h1)] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t: t[0])
    f.close
    return(listeDistances[:k])


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    ApprentissageTexture()
    ApprentissageColor()
    if request.method == 'POST':
        option = request.form['descriptor']

        file = request.files['query_img']
        #print(file)
        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        chemin = uploaded_img_path #Dharba scotch
        k = 3
        if option == "Texture":
            listeImage = RessemblaceImageTexture(chemin,k)
        if option == "Color":
            listeImage = RessemblaceImageColor(chemin,k)
        chemin = chemin.split("/")
        classe = (chemin[len(chemin)-1]).split("_")
        classe = classe[0]
        nbTrouve = 0
        scores=[]
        for i in range(len(listeImage)):
            scores.append((listeImage[i][0],"static/img/" + listeImage[i][1]))

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               descriptor = option,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")

