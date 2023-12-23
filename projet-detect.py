##############################################################################
############################# Installation ####################################
##############################################################################
# pip install imageai
# pip install object-detection
# pip install tensorflow
# pip install tensorflow-datasets
# pip install torch
# pip install torchvision

# Pour le lancer le script : python projet-detect.py


##############################################################################
############################# Importation ####################################
##############################################################################

import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import tkinter.messagebox

from PIL import Image, ImageTk
import PIL.Image
from PIL import ImageGrab

from imageai.Detection import ObjectDetection
import tensorflow_datasets as tfds

import string
import random
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from random import randrange



##############################################################################
########################## Methode Upload File ###############################
##############################################################################

def upload_image():
    global panelA  # declaration d'une variable globale

    f_types = [('Jpeg Files', '*.jpeg'), ('Jpg Files', '*.jpg'),
               ('PNG Files', '*.png')]  # declaration des types de fichier autorise

    filename = tk.filedialog.askopenfilename(
        multiple=True, filetypes=f_types)  # selectionne un ou plusieur fichiers
    col = 1  # commencer a partir de la colonne 1
    row = 3  # commencer a partir de la cellule 3
    for f in filename:
        image = PIL.Image.open(f)  # ouverture du fichier
        # redimensionne l'image originale
        resized_image = image.resize((200, 200), PIL.Image.ANTIALIAS)
        resized_image.save('saveUpload.png')  # sauvegarde de l'image
        photo1 = ImageTk.PhotoImage(resized_image)
        canvas = tk.Canvas(root, width=image.size[0], height=image.size[1])
        # creation d'une image
        canvas.create_image(0, 0, anchor=tk.NW, image=photo1)
        panelA = Label(image=photo1)
        panelA.image = photo1
        panelA.pack(side="left", padx=10, pady=10)
        panelA.configure(image=photo1)
        panelA.image = photo1  # affichage de l'image choisi par l'utilisateur

        if(col == 3):
            row = row+1
            col = 1
        else:
            col = col+1


def upload_image2():
    global panelB
    f_types = [('Jpeg Files', '*.jpeg'), ('Jpg Files', '*.jpg'),
               ('PNG Files', '*.png')]
    filename = tk.filedialog.askopenfilename(multiple=True, filetypes=f_types)
    col = 1
    row = 3
    for f in filename:
        image = PIL.Image.open(f)
        resized_image = image.resize((200, 200), PIL.Image.ANTIALIAS)
        resized_image.save('saveUp2.png')
        photo2 = ImageTk.PhotoImage(resized_image)
        canvas = tk.Canvas(root, width=image.size[0], height=image.size[1])
        canvas.create_image(0, 0, anchor=tk.NW, image=photo2)

        panelB = Label(image=photo2)
        panelB.image = photo2
        panelB.pack(side="left", padx=10, pady=10)
        panelB.configure(image=photo2)
        panelB.image = photo2

        if(col == 3):
            row = row+1
            col = 1
        else:
            col = col+1

##############################################################################
########################## Methode detect Matches ############################
##############################################################################


def matching():
    global panelC
    img1 = cv2.imread('saveUpload.png')  # lecture de l'image 1 telecharche
    img2 = cv2.imread('saveUp2.png')  # lecture de l'image 2 telecharche
    orb = cv2.ORB_create()  # creation d'un ORB

    #convertir les images au niveau de gris
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #definir les erreur de comparaison entre les 2 images
    def mesErr(img1, img2):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff**2)
        mse = err/(float(h*w))
        return mse, diff

    error, diff = mesErr(img1, img2)

    kp0, des0 = orb.detectAndCompute(img1, None)
    kp1, des1 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des0, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(
        img1, kp0, img2, kp1, matches[:25], img2,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    if (error < 30):
        img1 = cv2.imread('saveUpload.png') 
        img2 = cv2.imread('saveUp2.png') 
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig("match.png")  # sauvegarde de l'image match

    else:    
        tkinter.messagebox.showinfo("Correspondance","Il y a des matchs mais pas de reel correspondance")
        img1 = cv2.imread('saveUpload.png')
        img2 = cv2.imread('saveUp2.png') 
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig("match.png")  # sauvegarde de l'image match

    image = PIL.Image.open("match.png")  # ouverture de l'image match

    resized_image = image.resize((200, 200), PIL.Image.ANTIALIAS)
    # sauvegarde de l'image match redimensionne
    resized_image.save('match.png')
    photo = ImageTk.PhotoImage(resized_image)

    canvas = tk.Canvas(root, width=image.size[0], height=image.size[1])
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    panelC = Label(image=photo)
    panelC.image = photo
    panelC.pack(side="left", padx=10, pady=10)
    panelC.configure(image=photo)
    panelC.image = photo  # affichage de l'image match


##############################################################################
########################## Methode detection #################################
##############################################################################

def detection():
    global panelD  # declaration d'une variable global

    # instanciation d'un objet de la classe ObjectDetection
    recognizer = ObjectDetection()

    # definition des chemins necessaires
    path_model = "retinanet.pth"
    path_input = "saveUpload.png"
    path_output = "newimage1.jpg"

    # definition du model du jeu de donne Retinanet
    recognizer.setModelTypeAsRetinaNet()

    recognizer.setModelPath(path_model)  # chemin du model
    recognizer.loadModel()  # chargement du model
    # appel de la fonction detectObjectsFromImage()
    recognition = recognizer.detectObjectsFromImage(
        input_image=path_input,
        output_image_path=path_output
    )

    image = PIL.Image.open(path_output)  # ouverture de l'image telecharger 1
    photo = ImageTk.PhotoImage(image)

    canvas = tk.Canvas(root, width=image.size[0], height=image.size[1])
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    panelD = Label(image=photo)
    panelD.image = photo
    panelD.pack(side="left", padx=10, pady=10)
    panelD.configure(image=photo)
    panelD.image = photo
    # pour chaque element present dans la classe recognition --> element place dans le fichier
    for eachItem in recognition:
        mot = eachItem["name"].upper()
        write_yaml = "output.yaml"
        dict_file = [mot]

        with open(write_yaml, 'a') as yaml_file:
            yaml.dump(dict_file, yaml_file)


def detection2():
    global panelE

    recognizer = ObjectDetection()

    path_model = "retinanet.pth"
    path_input = "saveUp2.png"
    path_output = "newimage2.jpg"

    recognizer.setModelTypeAsRetinaNet()
    recognizer.setModelPath(path_model)
    recognizer.loadModel()
    recognition = recognizer.detectObjectsFromImage(
        input_image=path_input,
        output_image_path=path_output
    )

    image = PIL.Image.open(path_output)
    photo = ImageTk.PhotoImage(image)

    canvas = tk.Canvas(root, width=image.size[0], height=image.size[1])
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    panelE = Label(image=photo)
    panelE.image = photo
    panelE.pack(side="left", padx=10, pady=10)
    panelE.configure(image=photo)
    panelE.image = photo
    # pour chaque objet trouver dans l'image le nom est inscrit dans un fichier yaml pour les mots fleche
    for eachItem in recognition:
        mot = eachItem["name"].upper()
        write_yaml = "output.yaml"
        dict_file = [mot]

        with open(write_yaml, 'a') as yaml_file:
            # inscription des noms dans le fichier
            yaml.dump(dict_file, yaml_file)


##############################################################################
########################## Partie mot Fleche ############################
##############################################################################

motSelect = ''
precedent = [0, 0]
route = [0, 0]


def jeuMenu():
    jeuMenu = ttk.Frame(root)
    jeuMenu.pack(fill=tk.X, side=tk.TOP)

    titreMenu = tk.Label(jeuMenu,
                         text='Jeu de mot fleche',
                         font=('Helvetica', 23, 'bold'),
                         fg='#1BBDC5')
    titreMenu.pack(expand=True, fill=tk.X, pady=12)


def jeuStart():
    frame1 = tk.Frame(master=root)
    frame1.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=20, pady=12)

    frame2 = tk.Frame(master=root)
    frame2.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=10, pady=12)

    frame3 = tk.Frame(master=root)
    frame3.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True, padx=20, pady=30)

    wordList = []

    # ouverture du fichier cree avec les mots des images

    with open(r'output.yaml') as file:
        wordFile = yaml.load(file, Loader=yaml.FullLoader)
        wordList = [word for word in wordFile['words']]

    numWords = 4  # numero de mots a trouver
    size = 8  # taille de la grille

    arr = [[0 for x in range(size)] for y in range(size)]
    button = [[0 for x in range(size)] for y in range(size)]
    check = [0 for numWords in range(size)]
    dictionary = [0 for createWordSet in range(numWords)]

    # position possible des lettres aleatoire
    directionArr = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1],
                    [0, -1], [1, -1]]

    class Position:
        status = False  # si la case correspond a l'un des mots elle changera de couleur
        filled = False  # savoir si la case contient une lettre ou non
        char = ''  # lettre genere

    # remplir les cases
    def fill(x, y, word, direction):
        for i in range(len(word)):
            arr[x + direction[0] * i][y + direction[1]
                                      * i].char = word[i]  # ajout d'une lettre
            # la case contient une lettre donc filled = true
            arr[x + direction[0] * i][y + direction[1] * i].filled = True

    # remplir les cases avec les mots choisi
    def wordPlace(j, dictionary):
        # choix des mots de facon aleatoire avec random.choice()
        word = random.choice(wordList)
        # choix de la direction du mot de facon aleatoire
        direction = directionArr[random.randrange(0, 7)]

        x = random.randrange(0, size - 1)
        y = random.randrange(0, size - 1)

        # position du mot
        if (x + len(word) * direction[0] > size - 1
            or x + len(word) * direction[0] < 0
            or y + len(word) * direction[1] > size - 1
            ) or y + len(word) * direction[1] < 0:
            wordPlace(j, dictionary)
            return

        for i in range(len(word)):
            if (arr[x + direction[0] * i][y +
                                          direction[1] * i].filled == True):
                if (arr[x + direction[0] * i][y + direction[1] * i].char !=
                        word[i]):
                    wordPlace(j, dictionary)
                    return
        dictionary[j] = word

        # label des lettres des mots a chercher
        check[j] = tk.Label(frame2,
                            text=word,
                            height=1,
                            width=15,
                            font=('None %d ' % (10)),
                            fg='black',
                            bg='#7ED7E2',
                            anchor='c')
        check[j].grid()

        fill(x, y, word, direction)
        return dictionary

    # fonction pour colorer les mots selectionne
    def couleurMot(motSelect, valid):
        route[0] *= -1
        route[1] *= -1
        for i in range(len(motSelect)):
            if valid == True or arr[precedent[0] +
                                    i * route[0]][precedent[1] +
                                                  i * route[1]].status == True:
                button[precedent[0] + i * route[0]][precedent[1] +
                                                    i * route[1]].config(
                    bg='#7ED7E2',
                    fg='black')
                arr[precedent[0] + i * route[0]][precedent[1] +
                                                 i * route[1]].status = True
            elif (arr[precedent[0] +
                      i * route[0]][precedent[1] +
                                    i * route[1]].status == False):
                button[precedent[0] + i * route[0]][precedent[1] +
                                                    i * route[1]].config(
                    bg='black',
                    fg='#1BBDC5')

    # fonction pour verifier le mot dans la liste
    def verifMot():
        global motSelect

        if motSelect in dictionary:
            check[int(dictionary.index(motSelect))].configure(
                font=('Helvetica', 1), fg='#f0f0f0', bg='#f0f0f0')
            check[int(dictionary.index(motSelect))].grid()
            dictionary[dictionary.index(motSelect)] = ''

            couleurMot(motSelect, True)
        else:
            couleurMot(motSelect, False)
        motSelect = ''
        precedent = [0, 0]

    def lettreSelect(x, y):
        global motSelect, precedent, route
        newPressed = [x, y]

        if (len(motSelect) == 0):
            precedent = newPressed
            motSelect = arr[x][y].char
            button[x][y].configure(bg='#2BD5B4', fg='black')

        elif (len(motSelect) == 1 and (x - precedent[0])**2 <= 1
              and (y - precedent[1])**2 <= 1 and newPressed != precedent):
            motSelect += arr[x][y].char
            button[x][y].configure(bg='#2BD5B4', fg='black')

            route = [x - precedent[0], y - precedent[1]]
            precedent = [x, y]

        elif (len(motSelect) > 1 and x - precedent[0] == route[0]
              and y - precedent[1] == route[1]):
            motSelect += arr[x][y].char
            button[x][y].configure(bg='#2BD5B4', fg='black')
            precedent = [x, y]

    for x in range(size):
        for y in range(size):
            arr[x][y] = Position()

    for i in range(numWords):
        wordPlace(i, dictionary)

    for y in range(size):
        for x in range(size):

            if (arr[x][y].filled == False):
                arr[x][y].char = random.choice(string.ascii_uppercase)

            button[x][y] = tk.Button(
                frame1,
                text=arr[x][y].char,
                bg='black',
                fg='#1BBDC5',
                width=2,
                height=1,
                relief=tk.FLAT,
                command=lambda x=x, y=y: lettreSelect(x, y))
            button[x][y].grid(row=x, column=y)

    verifMotButton = tk.Button(frame2,
                               text="Vérifier le mot",
                               height=1,
                               width=15,
                               anchor='c',
                               bg="black",
                               font=('Helvetica', 10),
                               fg='white',
                               command=verifMot)
    verifMotButton.grid()


def main():
    jeuMenu()

    frame = tk.Frame(root)
    frame.pack(pady=56, padx=180)

    def updateUserInput():
        jeuStart()
        frame.destroy()

    tk.Button(frame,
              text="Commencer le jeu ",
              font=('Helvetica', 12),
              bg='#7ED7E2',
              fg='black',
              command=updateUserInput).grid(row=3,
                                            column=1,
                                            pady=8,
                                            ipady=6,
                                            ipadx=10)

    root.mainloop()


##############################################################################
########################## Fenetre principale ################################
##############################################################################

root = tk.Tk()
root.title("Projet Informatique Graphique et Vision")
root.geometry("770x550")  # Taille de la fenetre
tabControl = ttk.Notebook(root)

my_font1 = ('times', 18, 'bold')

titre = Label(root, text="Detection d'elements",
              font=('Helvetica', 23, 'bold'), fg='#1BBDC5')

consigne = Label(root, text="Pour televerser une image : Appuyer sur le bouton Image 1 ou Image 2 ",
              font=('Helvetica', 10), fg='#1BBDC5')

consigne2 = Label(root, text="Pour voir les correspondances : Appuyer sur le bouton Matching",
              font=('Helvetica', 10), fg='#1BBDC5')

consigne3 = Label(root, text="Pour detecter les elements de l'image : Appuyer sur le bouton Detect 1 (image 1) ou Detect 2 (image 2)",
              font=('Helvetica', 10), fg='#1BBDC5')

consigneImages = Label(root, text="La detection marchent pour les images contenant des : voitures, animaux, certains fruits (ex : banane,orange,pomme) ",
              font=('Helvetica', 10), fg='#1BBDC5')

consigneImages2 = Label(root, text="La detection ne marchent pas si l'element n'est pas present pas le jeu de donnee !",
              font=('Helvetica', 10), fg='#1BBDC5')
consigne4 = Label(root, text="Pour jouer : Appuyer sur le bouton Jouer",
              font=('Helvetica', 10), fg='#1BBDC5')
consigne5 = Label(root, text="Un conseil, appuyer sur le bouton Effacer avant de jouer !",
              font=('Helvetica', 10), fg='#1BBDC5')

# creation des boutons 
buttonImg1 = Button(master=root, command=upload_image, height=2,
                    width=15, text="Image 1", bg="#7ED7E2", relief=FLAT)
buttonImg2 = Button(master=root, command=upload_image2, height=2,
                    width=15, text="Image 2", bg="#7ED7E2", relief=FLAT)
buttonMatch = Button(master=root, command=matching, height=2,
                     width=15, text="Matching", bg="#7ED7E2", relief=FLAT)
buttonDetect1 = Button(master=root, command=detection, height=2,
                       width=15, text="Detect Image 1", bg="#7ED7E2", relief=FLAT)
buttonDetect2 = Button(master=root, command=detection2, height=2,
                       width=15, text="Detect Image 2", bg="#7ED7E2", relief=FLAT)
buttonJouer = Button(master=root, command=main, height=2,
                     width=15, text="Jouer", bg="#7ED7E2", relief=FLAT)
buttonQuit = Button(root, text="Quitter", height=2, width=10,
                    command=root.destroy, bg="#7ED7E2", relief=FLAT)

# placement des boutons dans la fenetre
buttonImg1.place(x=0, y=210)
buttonImg2.place(x=115, y=210)
buttonMatch.place(x=230, y=210)
buttonDetect1.place(x=345, y=210)
buttonDetect2.place(x=460, y=210)
buttonJouer.place(x=575, y=210)
buttonQuit.place(x=690, y=210)
titre.pack()
consigne.place(x=5, y=60)
consigne2.place(x=5, y=80)
consigne3.place(x=5, y=100)
consigne4.place(x=5, y=120)
consigne5.place(x=5, y=140)

consigneImages.place(x=5, y=160)
consigneImages2.place(x=5, y=180)


def effacer():
    panelA.pack_forget()
    panelB.pack_forget()
    panelC.pack_forget()
    panelD.pack_forget()
    panelE.pack_forget()


bouton = Button(root, text="Effacer", command=effacer,
                height=2, width=15, bg="#46858C", relief=FLAT)


bouton.place(x=0, y=250)

tabControl.pack(expand=1, fill="both")

root.mainloop()  # garde la fenetre de l'interface ouverte
