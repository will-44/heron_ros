# Manipulabilité du bras au fil de l'approche

Analyse temporelle de la manipulabilité de Yoshikawa `w = sqrt(det(J Jᵀ))`, en complément
des moyennes par phase de `cmp_manip.png` (setup 1 et setup 2).

**Pourquoi.** Les moyennes par phase mélangent des phases de durées très différentes :
l'approche de dépose dure 90 s à base fixe contre 49 s à base mobile (setup 2). Une
moyenne plus élevée peut donc simplement vouloir dire « le bras est resté plus longtemps
garé dans sa posture finale ». La colonne `manipulability` de `joint_states.csv` est
journalisée à ~106 Hz (≈ 500 000 échantillons par run) : il n'y a aucune raison de s'en
tenir à une valeur par phase.

**Méthode.** `manip_over_time.py` charge les quatre runs avec leur propre classification
(les épisodes écartés n'entrent dans aucun calcul, cf. la règle des rapports), aligne
chaque approche sur son propre début, et produit :

- `figures/manip_over_time.png` — médiane et intervalle interquartile de `w(t)`, une
  ligne par condition, deux lignes de panneaux (saisie, dépose) × deux setups ;
- `figures/manip_dynamics.png` — trois métriques par approche : temps de **stabilisation**
  (instant après lequel `w` ne repasse plus sous 80 % de sa valeur finale), `w` **à
  l'arrivée** (médiane des 3 dernières secondes), **temps passé sous `w = 0,05`**
  (bande basse — voir l'avertissement ci-dessous, ce n'est *pas* une quasi-singularité) ;
- `figures/manip_by_outcome.png` — approche de dépose, cycles réussis contre cycles
  finissant en collision ;
- `tables/manip_legs.csv` — une ligne par approche, `tables/manip_summary.csv|md` — les
  médianes par condition et le test de Mann-Whitney associé.

Reproduire : `python3 analysis/manipulability/manip_over_time.py` depuis `heron_experiment/`. Sorties dans `results/analysis/manipulability/`.

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

## Ce que l'analyse temporelle montre

### 1. La moyenne par phase est biaisée par la durée

Setup 2, approche de dépose : la base fixe affiche la meilleure moyenne (0,254 contre
0,180) et la meilleure valeur à l'arrivée (0,273 contre 0,223), mais elle met 90 s à y
arriver contre 49 s. Le bras finit effectivement tendu vers la cible — c'est une posture
bien conditionnée — seulement il y parvient tard, en passant par un creux dans la bande
basse de `w` que la base mobile évite.

| Setup 2 — approche de dépose | base fixe | base mobile | p (Mann-Whitney) |
|---|---|---|---|
| durée de la phase | 89,9 s | 48,9 s | 0,000 |
| stabilisation (80 % de w final) | 31,8 s | 22,0 s | 0,001 |
| temps dans la bande basse w < 0,05 | 8,7 s | 0,0 s | 0,001 |
| temps réellement quasi-singulier w < 0,01 | **0,0 s** | **0,0 s** | 0,68 |
| w minimal sur la phase | 0,032 | 0,053 | 0,020 |
| w à l'arrivée | 0,273 | 0,223 | 0,000 |

Même structure au setup 1 : stabilisation 30,3 s contre 18,4 s (p = 0,018) pour une durée
de phase de 67 s contre 42 s, avec des `w` à l'arrivée équivalents (0,217 contre 0,210).

> **Avertissement sur le seuil, et correction d'une formulation antérieure.** Une version
> précédente de cette note, des rapports et de la présentation décrivait ces 8,7 s comme
> du temps « en quasi-singularité ». C'est faux et il faut le retirer. Sur les 1 596 105
> échantillons des quatre runs, **34 % sont sous w = 0,05**, et la posture de repos du
> bras elle-même se situe à w = 0,012–0,020, donc *sous* le seuil. Surtout, pendant
> l'approche de dépose, le **minimum** de w vaut 0,032 (fixe) et 0,053 (mobile) : aucune
> des deux conditions ne descend sous 0,02, ni a fortiori sous 0,01. Le temps
> effectivement quasi-singulier y est de **0,0 s des deux côtés**. L'écart mesuré est réel
> et significatif (9,5 s contre 0 s, soit 7,7 % de la phase, p = 0,001, δ de Cliff = 0,66),
> mais c'est un **creux dans une bande que le bras occupe couramment**, pas un frôlement
> de singularité. Seule l'approche de saisie va réellement bas (§3).

### 2. En scène encombrée, la base mobile améliore nettement la posture de saisie

C'est le résultat que la moyenne par phase ne laissait pas voir, et il nuance la slide
« la saisie ne gagne rien » (fondée sur Q et sur le temps) :

| Setup 2 — approche de saisie | base fixe | base mobile | p |
|---|---|---|---|
| w à l'arrivée | 0,085 | 0,121 (**+42 %**) | 0,000 |
| w médian sur la phase | 0,053 | 0,073 | 0,037 |

Sur table vide l'effet s'inverse mais reste marginal (0,233 contre 0,214, soit −8 %) :
la pose initiale y est déjà bonne, il n'y a rien à gagner.

### 3. L'approche de saisie, elle, passe réellement par une quasi-singularité

C'est la seule phase où le terme se justifie, et il se justifie avec un critère strict :
le temps passé sous **w = 0,01** y vaut 1,4 à 2,3 s selon le run (3 à 7 % de la phase),
et le minimum atteint 0,000 dans la plupart des approches — contre 0,0 s partout sur
l'approche de dépose. La cause est géométrique et identifiée depuis :
[`../acceleration/`](../acceleration/README.md) §4 montre que q₅ balaie
−73° → 0 → +38° entre 12 et 18 s, c'est-à-dire que le poignet du M1013 traverse sa
singularité à chaque approche de saisie.

Les deux conditions se superposent sur ce segment (1,37 s contre 1,45 s au setup 1,
p = 0,43 ; 2,31 s contre 1,89 s au setup 2, p = 0,12) : c'est une propriété de la
trajectoire de sortie de la pose de repos, pas du positionnement de la base. C'est le
point à traiter indépendamment du contrôleur (pose de repos, ou contrainte sur q₅).

### 4. La manipulabilité ne prédit pas les échecs

Sur l'approche de dépose, les cycles finissant en collision n'ont pas une manipulabilité
plus basse que les cycles réussis — au setup 2 à base fixe, 0,257 contre 0,273 à
l'arrivée, avec les mêmes courbes à quelques pourcents près. Les échecs viennent de
l'atteignabilité de la pose de dépose à travers l'encombrement (contrainte anti-collision
du MPC active, erreur FK finale de 0,20–0,59 m), pas d'un bras mal conditionné. La
manipulabilité mesure la qualité de la posture *atteinte*, elle ne dit rien de ce que le
MPC n'arrive pas à atteindre — ce que mesure Q.

## Limites

- `w` de Yoshikawa mélange les unités de translation et de rotation : les valeurs
  absolues ne sont comparables qu'entre postures de ce même bras.
- C'est un scalaire isotrope : il ne dit pas si la mobilité restante est orientée vers la
  cible. Une manipulabilité directionnelle (ellipsoïde projeté sur l'axe d'approche)
  demanderait de recalculer la jacobienne depuis l'URDF du M1013 — non fait ici.
- Les courbes sont tronquées là où moins de 5 approches restent actives, ce qui explique
  que les lignes « base mobile » s'arrêtent plus tôt : ces approches sont plus courtes.
