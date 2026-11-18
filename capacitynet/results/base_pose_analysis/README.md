# Pourquoi la base ne se met pas devant la cible

Question posée : la base mobile semble s'arrêter avec la cible à ~30° sur le côté plutôt
que droit devant. Impression, manque de temps, autre extremum, ou vraie propriété ?

Reproduire : `python3 base_pose_analysis/bearing_over_time.py` depuis `results/`.
Sorties : `figures/bearing_over_time.png`, `tables/polar_legs.csv`,
`tables/polar_summary.csv|md`, `tables/gradient_direction.csv`, `tables/stop_reasons.csv`.

**Mesure.** `samples.csv` journalise le but MPC dans `dsr01/world`, le repère lié à la
base du bras, donc `azimut = atan2(goal_y, goal_x)` (0° = cible droit devant) et
`distance = hypot(goal_x, goal_y)` sont les coordonnées polaires de la cible telle que le
bras la voit. Contrôle de cohérence du repère : sur les runs à base fixe l'azimut doit
être plat — il l'est, à 0,0–0,4° près sur toute la phase.

## 1. Ce n'est pas une impression

| | azimut au début | azimut à la fin | Δ | distance début → fin | Q début → fin |
|---|---|---|---|---|---|
| S1 dépose, base mobile | 44,6° | **30,3°** | −14,3° | 0,98 → 0,90 m | 0,73 → 0,84 |
| S2 dépose, base mobile | 47,8° | **35,2°** | −13,9° | 1,09 → 0,93 m | 0,57 → 0,74 |
| S1 saisie, base mobile | 56,8° | 47,7° | −8,3° | 0,77 → 0,76 m | 0,79 → 0,81 |
| S2 saisie, base mobile | 34,3° | 34,3° | −0,3° | 0,60 → **0,75 m** | 0,82 → 0,86 |
| (base fixe, les 4 jambes) | — | inchangé | ±0,4° | inchangée | — |

L'observation est juste : sur les approches de dépose, la base réduit franchement
l'azimut — de 14° — et s'arrête autour de **30–35°**. Sur l'approche de saisie elle ne le
réduit presque pas.

## 2. Sur la dépose : ce n'est pas un optimum, la base est arrêtée en pleine descente

Trois éléments concordants.

**Le gradient pointe encore vers la cible au moment où le mouvement s'arrête.** En
projetant ∇Q sur la direction de la cible et sur sa perpendiculaire :

| approche de dépose | sur toute la jambe | aux 3 derniers cycles |
|---|---|---|
| composante « se mettre en face » (S1) | +0,84 (positive dans 85 % des cycles) | **+0,80** |
| composante « se mettre en face » (S2) | +0,69 (positive dans 89 % des cycles) | **+0,57** |
| ‖∇Q‖ (S1 / S2) | 0,200 / 0,286 | 0,201 / 0,241 |

La norme du gradient à l'arrêt est celle du milieu de jambe : rien n'indique un point
stationnaire.

**La base ne s'arrête jamais parce que le gradient a convergé.** Sur les 42 jambes de
dépose des deux runs mobiles : 30 arrêts sur *arrival latch* (les sphères touchent, le
bras est arrivé), 12 sur interlock, **0 sur convergence du gradient**. Sur les jambes de
saisie, 3 convergences sur 45.

**Le budget de déplacement est trop court d'un facteur ~2.** La commande est ouverte 19 à
21 s par jambe, à 0,020 m/s strictement (bang-bang), soit **0,40 m** de trajet — ce que le
chemin commandé mesuré confirme (0,39–0,41 m médians). Or passer de 45° à 0° à 0,9 m de
rayon demande un arc de **0,71 m**, et il reste encore 0,47 m à faire depuis les 30°
atteints, c'est-à-dire plus qu'une jambe entière. Sur `bearing_over_time.png`, l'azimut
plafonne vers 20 s : exactement l'instant où la commande se ferme, pas un palier du
critère.

Donc, pour la dépose : **manque de temps, pas autre extremum.**

## 3. Sur la saisie : là, c'est une vraie propriété — et elle n'est pas celle qu'on croit

Le diagnostic s'inverse. Aux derniers cycles, la composante « se mettre en face » du
gradient est **négative** (−0,07 au setup 1, −0,42 au setup 2) : le modèle ne demande pas
de se mettre devant. Ce qu'il demande est **radial**. Au setup 2 la composante radiale
médiane vaut **−0,93** sur toute la jambe : la base part à 0,60 m, c'est trop près, et Q
s'améliore en **reculant** — ce qu'elle fait, 0,60 → 0,75 m, pour un gain de Q de 0,82 à
0,86. L'azimut de 34° n'est pas une descente inachevée, il n'est simplement pas l'objet
de l'optimisation ici.

## 4. Ce que gouverne Q, dans la plage réellement visitée

À distance fixée, Q varie peu avec l'azimut ; à azimut fixé, il varie beaucoup avec la
distance (échantillons des quatre runs) :

| distance | Q médian |
|---|---|
| 0,55–0,75 m | 0,83 |
| 0,75–0,95 m | 0,76 |
| 0,95–1,15 m | 0,52–0,65 |

contre 0,75 à 0,79 sur toute la plage d'azimut 20–60° à distance 0,75–0,95 m. **Le levier
dominant est la distance, l'azimut est un effet de second ordre** dans la plage explorée.

## 5. Deux limites structurelles, dont une qui est le vrai levier

- **La base ne tourne jamais.** `wz` vaut exactement 0 dans les 105 137 messages
  `/cmd_vel` des deux runs mobiles. Le contrôleur ne commande que de la translation, et
  `ObstacleMapTransformer` n'évalue que des cartes d'obstacles **translatées** : il n'y a
  pas de gradient en rotation, le modèle ne peut même pas savoir si tourner aiderait. Or
  le lacet est de très loin le moyen le moins cher de changer l'azimut : 30° de rotation
  à 0,2 rad/s coûtent **2,6 s**, contre 0,47 m de translation latérale soit **24 s** au
  plafond de 0,02 m/s. C'est le premier levier à essayer.
- **On ne sait pas si 0° est meilleur, parce que la base n'y est jamais allée.** Sur
  l'ensemble des quatre runs, le 1ᵉʳ centile de |azimut| vaut 21° : aucune donnée en
  dessous. Affirmer « le bras préfère travailler à 30° » n'est pas soutenu ; ce qui est
  établi, c'est que dans la plage 20–60° l'azimut ne fait pas grand-chose à Q, et que la
  descente vers 0° est interrompue avant d'aboutir sur la dépose.

## Comment trancher pour de bon

Un balayage hors ligne : reprendre une grille de voxels enregistrée avec son but, et
évaluer le modèle sur un éventail de poses de base — translation **et** lacet — pour
tracer Q(azimut, distance) au-delà de la plage visitée. C'est la seule façon de savoir si
0° est un maximum, un plateau, ou moins bon que 30°. Non fait ici : il faut une grille de
voxels enregistrée, que ces CSV ne contiennent pas.
