# Accélérations articulaires

Troisième volet après [`../manipulability_analysis/`](../manipulability_analysis/README.md)
et [`../velocity_analysis/`](../velocity_analysis/README.md) : mêmes runs, même règle
d'exclusion, même alignement sur le début de chaque approche.

Reproduire : `python3 acceleration_analysis/accel_over_time.py` depuis `results/`.
Sorties : `figures/accel_noise.png`, `figures/accel_over_time.png`,
`figures/accel_by_joint.png`, `figures/wrist_singularity.png`,
`tables/accel_noise_check.csv`, `tables/accel_legs.csv`, `tables/accel_summary.csv|md`,
`tables/accel_window_sensitivity.csv`.

## Lecture des courbes : aucune jambe n'est masquée

Les figures temporelles ne coupent plus les courbes à un quorum. La règle précédente
(`ok = n >= 5`, ne tracer que tant qu'il reste au moins 5 jambes) biaisait les deux bouts :
la moyenne tracée portait sur un groupe qui se vide et se **sélectionne** — les jambes
lentes sont aussi celles dont le score est encore mauvais — et la courbe s'arrêtait avant
la fin des jambes les plus longues. Désormais :

- chaque jambe est tracée jusqu'à sa propre fin, et l'axe des temps s'arrête avec la plus
  longue ;
- la médiane est **pleine** tant qu'au moins la moitié des jambes sont en cours, et
  **pointillée** ensuite : la partie pointillée reste vraie pour ces jambes-là, elle n'est
  simplement plus représentative du groupe ;
- l'intervalle interquartile n'est tracé que sur la partie pleine ;
- la marche fine en bas de chaque panneau donne le **nombre de jambes encore en cours**.

Code commun : [`../leg_survival.py`](../leg_survival.py).

## 0. Avertissement : la colonne `acc_joint_*` journalisée n'est pas exploitable

`workspace_reachability_node._on_joint_states` calcule `vel = Δpos/Δt` puis
`acc = Δvel/Δt` entre messages `/joint_states` consécutifs, dont le Δt est très
irrégulier : ~23 % des échantillons arrivent moins de 2 ms après le précédent,
~54 % autour de 10 ms, 3,5 % après un trou de 50–200 ms.

La vitesse survit à ce traitement, la double différence non :

| | autocorrélation à lag 1 de `acc_joint_*` | de l'accélération redérivée |
|---|---|---|
| setup 1 base fixe | **−0,59** | +0,99 |
| setup 1 base mobile | **−0,60** | +0,99 |
| setup 2 base fixe | **−0,58** | +0,99 |
| setup 2 base mobile | **−0,57** | +0,99 |

Une accélération physique échantillonnée à 100 Hz a une autocorrélation proche de
+0,99 ; −0,58 est la signature d'une suite de bruit à signe alterné. `figures/accel_noise.png`
montre les deux signaux côte à côte : ±10 rad/s² de hachures d'un côté, ±0,8 rad/s²
structurés de l'autre.

**Conséquence directe : tous les chiffres d'accélération produits jusqu'ici à partir de
cette colonne — max, min, moyenne, écart-type, y compris ceux de la présentation —
mesurent la gigue d'horodatage du logger, pas le bras.** Le maximum en particulier est
sans aucun contenu physique (jusqu'à 27 rad/s² observés). Deux corrections possibles :
redériver a posteriori comme ici, ou, mieux, faire publier au nœud sa propre horloge
régulière (ou lire les champs `velocity`/`effort` du driver) pour les runs suivants.

*À noter aussi : `vel_joint_*` est également une différence finie de la position, pas une
valeur rapportée par le driver — contrairement à ce qui était écrit dans la note sur les
vitesses, maintenant corrigée. La vitesse reste fiable (autocorrélation +0,98) ; seules
ses valeurs extrêmes ponctuelles sont à prendre avec réserve.*

## Méthode

Positions rééchantillonnées sur une grille uniforme à 100 Hz, les segments couvrant un
trou de plus de 50 ms dans le log sont masqués, puis dérivation de Savitzky-Golay
(ordre 3, fenêtre **0,41 s**), et les bords de filtre sont écartés. Les valeurs absolues
dépendent de la fenêtre (une fenêtre de 0,21 s donne des amplitudes 2,5 fois plus
grandes) : **seules les comparaisons à fenêtre fixe ont un sens**. Le rapport base
fixe / base mobile a été vérifié à 0,21 / 0,41 / 0,61 s (`accel_window_sensitivity.csv`) :
il est stable.

On y ajoute le **SPARC** (spectral arc length), une mesure de douceur sans dimension
calculée sur le profil de ‖q̇‖ et indépendante de la fenêtre de filtrage.

## Ce que l'analyse montre

### 1. L'amplitude des accélérations ne distingue pas les deux conditions

Comme pour les vitesses : moyenne de ‖q̈‖ 0,12–0,17 rad/s², p95 0,48–0,64, aucune
différence significative sur les quatre comparaisons (p ≥ 0,12), à toutes les fenêtres.
Le MPC applique le même régime d'accélération quelle que soit la base.

### 2. Mais à base fixe, le bras continue d'accélérer longtemps après

`accel_over_time.png`, panneau « table encombrée — approche de dépose » : la base mobile
retombe sous 0,05 rad/s² vers 40 s, la base fixe reste entre 0,10 et 0,30 rad/s² jusqu'à
60 s et au-delà. Ce n'est pas une dérive lente, ce sont des **ré-accélérations
répétées** — le MPC relance le bras en permanence.

En comparant à durée appariée (on ne garde que les approches dont la durée existe dans
les deux conditions, 9 contre 12 approches, durées médianes 63 et 54 s, p = 0,41) :

| Setup 2 — approche de dépose, durées appariées | base fixe | base mobile | p |
|---|---|---|---|
| moyenne de ‖q̈‖ | 0,182 | 0,153 | 0,002 |
| max de ‖q̈‖ | 2,12 | 1,62 | 0,036 |
| inversions d'accélération (1/s/axe) | 3,83 | **4,13** | 0,017 |

À durée égale, **la base fixe accélère plus fort, la base mobile fait plus de corrections
mais plus petites**. Le maximum de ‖q̈‖ est d'ailleurs significatif même sans appariement
(1,94 contre 1,56, p = 0,006), comme l'était le maximum de vitesse. Attention toutefois :
l'appariement laisse 9 et 12 approches, c'est une tendance à confirmer, pas un résultat
établi.

### 3. La différence de douceur (SPARC) est un artefact de durée — ne pas la citer

Le SPARC sépare très nettement les conditions sur l'approche de dépose (−5,65 contre
−3,15 au setup 2, p = 0,001 ; −3,71 contre −2,81 au setup 1, p = 0,039), ce qui se lit
spontanément comme « la base fixe produit un mouvement moins doux ». Mais le SPARC est
corrélé à la durée de la phase à **ρ = −0,84** (et −0,75 à −0,81 *à l'intérieur de chaque
condition*), et l'écart **disparaît dès qu'on apparie les durées** (p = 0,37). Une phase
plus longue contient plus de sous-mouvements, donc un SPARC plus négatif : la métrique
redit la durée. Même conclusion pour l'inversion observée sur l'approche de saisie au
setup 2 (base mobile apparemment moins douce, p = 0,004) : elle ne survit pas non plus à
l'appariement (p = 0,10). Résultat consigné ici pour éviter qu'il soit repris ailleurs.

### 4. Le résultat structurel : l'approche de saisie passe par la singularité de poignet

`accel_by_joint.png` : sur l'approche de saisie, **J5 est de loin l'axe le plus sollicité**
(0,087 rad/s² médian contre 0,030–0,063 pour les autres), alors que sur l'approche de
dépose la charge est répartie sur J1–J4 et que J5 est l'un des plus calmes (0,042).

La raison est dans `wrist_singularity.png` : pendant l'approche de saisie, q₅ balaie
−73° → 0 → +38°, et **la valeur minimale de |q₅| est de 0,0° en médiane par approche**,
dans les quatre runs. Le bras passe donc par q₅ = 0, la singularité de poignet du M1013,
à **chaque** approche de saisie, entre 12 et 18 s — exactement l'instant et la durée du
passage quasi-singulier relevé dans l'analyse de manipulabilité (w tombe à 0,02–0,03).
Les courbes des deux conditions se superposent : c'est une propriété de la trajectoire
repos → pré-saisie, pas du positionnement de la base. Sur l'approche de dépose, |q₅| ne
descend jamais sous ~20–50°, et il n'y a pas de passage quasi-singulier.

C'est le point le plus actionnable des trois analyses : le passage par q₅ = 0 est
gratuit — il coûte de la manipulabilité, de l'accélération sur le poignet et
probablement une part des blocages — et il se corrige en changeant la pose de repos ou
en contraignant q₅ dans la trajectoire de sortie, indépendamment du contrôleur de base.

## Limites

- Les amplitudes dépendent de la fenêtre de dérivation ; elles ne valent qu'en
  comparaison, à fenêtre fixe (voir `accel_window_sensitivity.csv`).
- Ce sont des accélérations **mesurées** sur la position rapportée, pas les accélérations
  commandées ; on ne peut pas en déduire directement une marge par rapport aux limites du
  driver, ni compter les commandes SpeedJ rejetées.
- Les comparaisons à durée appariée reposent sur 9 à 15 approches par condition.
- `wrist_singularity.png` trace |q₅| ; une vraie mesure de proximité de singularité
  demanderait la plus petite valeur singulière de la jacobienne, non calculée ici (même
  limite que l'analyse de manipulabilité).
