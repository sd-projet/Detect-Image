# Detect-Image

Cette application en python a pour but de pouvoir jouer à un jeu de mot fléchés.
Dans un premier temps, l’utilisateur devra téléverser au minimum 2 images à travers les
boutons Image 1 et Image 2 et appuyer sur detect 1 ou detect 2 (voir annexe 1).
Cela permettra alors d’appeler une fonction qui elle réalisera la détection. En effet, cette fonction va chercher les éléments présent sur la photo et noter leur noms dans un fichier yaml.
Ce fichier yaml servira à remplir les mots à chercher pour le jeu de mot fléché.
De plus, nous avons insérer des mots au cas où l’utilisateur n’entrerait pas assez d’image et qu’il n’y aurai pas assez de mot pour le jeu.
Par ailleurs, l’utilisateur peut également vérifier si les 2 images qu’il a téléversé ont des similitudes en appuyant sur le bouton matching.
Si les images ne se corresponde pas vraiment il y aura un message qui apparaitra.
