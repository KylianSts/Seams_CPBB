TO DO 

    - Réfléchir à l'optimisation des différents codes et à leur restructuration
    - Tout mettre bout à bout dans le run pipeline
    - Créer un yaml pour plus de facilité l'utilisation et de suivi



Comment les scores de tirs sont calculé et afficher ? --> ne calucler que quand on a un mouvement déscendant de la balle sur une trajectoir donc pas sur une  image (spatial trigger ou autre ?) --> ne pas prendre la moyenne des scores pour la détection du tir 

rajouter d'autres métriques et définri des règles avec ces métriques pour l'optimisation ou la détection d'évènement le faire par équipe et en total (rajouter un dashboard quelque part sur l'écran) (max et min vitesse par équipe --> posibilité du saut ? ne pas prendre un joueur pour le spacing si trop loin)

regarder le futur pour équipe éventuellement et pour savoir quand enlever le logo avant qu'il ne soit trop tard 

Critique : L'alpha blending en NumPy sur CPU pour du 1080p ou 4K peut devenir un léger goulot d'étranglement. Une opération équivalente via OpenCV (cv2.addWeighted) ou directement sur tenseur GPU serait plus rapide.

faire tourner en 15 frames (ou 20/25) au lieu de 30 et faire de l'interpolation interframe ?

refaire sortir les moments fort en utilisant l'audio

Hybridation avec une méthode peu gourmandeOui, se baser uniquement sur la couleur (HSV) est risqué si les deux équipes ont des maillots proches (ex: Bleu marine vs Noir) ou si la balance des blancs de la caméra est mauvaise.Voici deux méthodes "gratuites" (qui ne demandent aucune inférence de réseau de neurones supplémentaire) pour consolider les votes :Méthode A : Le "ReID Embedding" de BotSort (Le Graal)C'est l'astuce ultime. Pour que BotSort puisse traquer tes joueurs, il utilise un petit réseau de neurones (OSNet) qui extrait un "vecteur de caractéristiques" (embedding) de 512 dimensions pour chaque boîte. C'est comme une empreinte digitale visuelle qui encode la couleur, mais aussi la texture (rayures, logo, forme du short).L'idée : Puisque BotSort a déjà calculé ce vecteur pour chaque joueur à chaque frame, on peut le récupérer gratuitement ! On peut faire notre K-Means (ou GMM) sur ces vecteurs au lieu des histogrammes HSV. C'est immensément plus robuste.Méthode B : Le Vecteur de Vitesse (Direction spatiale)Au basket, en phase d'attaque placée, tous les joueurs de l'équipe offensive regardent vers le panier, et tous les défenseurs regardent vers le porteur de balle. Mais surtout, lors des transitions (Fast Break), les 5 joueurs d'une équipe sprintent dans la même direction ($V_x$).L'idée : Si ton lissage par couleur hésite pour un joueur qui sprinte vers la gauche, et que 4 autres joueurs identifiés "Verts" sprintent aussi vers la gauche, tu peux utiliser cette corrélation spatiale pour pondérer le vote.

Il y a des déplacement de balle totalement impossible. Bien qu'on ne peut pas utilsier l'homographie pour avoir sa vitesse exacte car elle n'est pas à y=0 il faut noté que en regardant le nombre de pixel de l'image et en regardant le nombre de pixel du ballon il y a des mouvements impossible (proche du panier et l'image d'après dans les mains d'un joueur à 3 points puis re vers le panier en une demi seconde) qui devrait indiqué un saut dans les prédictions

utiliser hauteur du panier pour la 3D 

optimisation de sam: juger la pertinence de yolo seg (possible entrainement si nécessaire en partant de SAM pour créer la donnée)

Rajotuer des données de trajectoir de balle nottament en utilisant nantes et cergy puis réentrainer le modèle sur 200 epochs à 1280
