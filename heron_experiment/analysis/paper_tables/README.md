# Les deux tableaux de l'article

`python3 analysis/paper_tables/paper_tables.py` depuis `heron_experiment/`. Sorties dans `results/analysis/paper_tables/tables/` :
`tables.tex` (les deux tableaux LaTeX remplis), `perf.csv`, `cost.csv`, `accel_pool.csv`.

Un seul script pour que chaque chiffre de l'article vienne du même endroit, avec une
seule règle d'exclusion et une définition explicite par ligne. Les définitions sont dans
la docstring du script ; l'essentiel ci-dessous.

## Ce qui était faux dans la version précédente du tableau

**Les durées étaient des médianes, pas des moyennes.** Les quatre valeurs de « Total run
time » (214 / 178 / 250 / 189) sont exactement les médianes ; les moyennes sont
224 / 184 / 261 / 195. Même chose pour les deux approches. L'écart n'est pas cosmétique :
la distribution des durées est à queue lourde d'un seul côté (une run qui traîne peut
durer trois fois la médiane, aucune ne peut durer moins que le temps de trajectoire
incompressible), donc moyenne > médiane partout, et de 10 à 20 % sur l'approche de dépose.

| ligne | ancienne valeur (médiane) | moyenne ± écart-type |
|---|---|---|
| Total, setup 1 fixe / mobile | 214 / 178 | 224 ± 50 / 184 ± 21 |
| Total, setup 2 fixe / mobile | 250 / 189 | 261 ± 62 / 195 ± 25 |
| Approche saisie, s1 / s2 | 41, 42 / 36, 39 | 42 ± 5, 41 ± 5 / 35 ± 4, 46 ± 20 |
| Approche dépose, s1 / s2 | 71, 42 / 105, 50 | 84 ± 41, 48 ± 19 / 127 ± 78, 54 ± 18 |

Deux conséquences à regarder avant de publier :

- **L'approche de saisie du setup 2 se creuse.** En médiane la base mobile est plus lente
  de 3,5 s (35,8 → 39,3) ; en moyenne elle l'est de 11 s (35,1 → 46,1), avec un écart-type
  de 20 s contre 4 s. Quatorze des dix-huit runs tiennent entre 28 et 45 s, les quatre
  autres durent 70, 71, 72 et 105 s et portent tout l'écart (moyenne 42,6 s sans la plus
  longue). C'est la ligne où moyenne ± écart-type informe le moins bien ; si le tableau
  reste en moyennes, elle mérite une note ou la médiane entre parenthèses.
- **L'approche de dépose du setup 2 fixe** passe de 105 à 127 ± 78 s. Les dix runs
  réussies s'étalent de 51 à 317 s : à base fixe la dépose réussit soit vite, soit après
  un très long tâtonnement, et rien ne s'agglutine autour de la médiane. L'effet « base
  mobile » y est plus grand en moyenne (127 → 54 s) qu'en médiane (105 → 50 s), donc la
  conclusion ne change pas de sens.

Le reste était déjà en moyennes et se confirme : les temps de cycle MPC et gradient, le
$R_i$ de fin d'approche, les accélérations. Les distances base–cible correspondaient bien
à la jambe de **dépose** (et non de saisie) ; la ligne de saisie a été ajoutée.

## Définitions

- **Conditions.** setup 1 = table vide (`setup_1/20260916_141632_base_fixe` fixe,
  `setup_1/20260916_110958_base_mobile` mobile) ; setup 2 = table encombrée (`setup_2/20260918_174739_base_fixe`
  fixe, `setup_2/20260921_100545_base_mobile` mobile). Chaque run est chargé par son propre loader, qui
  porte sa propre classification d'épisodes.
- **Exclusion.** `outcome_valid()` du run lui-même, jamais une réimplémentation : un
  épisode exclu (infrastructure, intervention opérateur, ou échec de préhension/lâcher,
  qui ne relève pas du contrôleur de base) n'entre dans aucune statistique. N vaut donc
  23 / 25 / 17 / 20 et diffère selon la condition — la légende du tableau doit le dire,
  un N unique serait faux.
- **Taux de réussite** = succès / runs valides.
- **Durées** : moyenne ± écart-type sur les runs **réussies**. Cycle complet depuis
  `episodes_index.csv`, phases d'approche depuis `phase_summary.csv`.
- **$R_i$ en fin d'approche** : réachabilité de la sphère du but telle que
  l'enregistreur la mesure, moyennée sur les **3 dernières secondes** de la phase
  (`samples.q_goal`, buts frais uniquement, `goal_age_s < 1`), puis moyenne ± écart-type
  sur les runs réussies. C'est la valeur sur laquelle le bras finit réellement, pas le
  dernier cycle du contrôleur.
- **Distance base–cible** : `hypot(goal_x, goal_y)` dans le repère du bras, au premier
  échantillon de la phase puis moyenné sur ses 3 dernières secondes, moyenné sur les
  runs réussies. Attention : c'est la moyenne des portées par run, alors que les figures
  de comparaison tracent la portée de la position **médiane** du but — deux nombres
  différents, proches ici (0,00 à 0,02 m d'écart).
- **Temps de cycle MPC et gradient** : tous les cycles journalisés de toutes les runs
  valides, réussies ou non — un coût se paie quelle que soit l'issue. Le dépassement de
  budget est recompté à partir de `solve_ms > 400`, pas de la colonne `budget_exceeded`.
- **Accélérations articulaires** : redérivées des positions journalisées (grille
  uniforme 100 Hz, Savitzky-Golay, ordre 3, fenêtre 0,41 s), parce que la colonne
  `acc_joint_*` du log est du bruit de différentiation — voir
  [`../acceleration/README.md`](../acceleration/README.md). Mises en
  commun sur tous les échantillons de toutes les **phases de mouvement** (les deux
  approches, les deux transferts, les deux retours home) de toutes les runs valides ;
  les périodes d'inactivité sont exclues. Les valeurs absolues dépendent de la fenêtre,
  la comparaison fixe/mobile non.

## Pourquoi le $R_i$ du tableau ne vaut pas la fin de la courbe de `cmp_q_legs.png`

Les deux mesurent la même grandeur — la réachabilité de la sphère du but — mais **pas au
même instant, pas depuis la même source, et pas sur la même population**. Setup 2, sphère
du but :

| | saisie fixe | saisie mobile | dépose fixe | dépose mobile |
|---|---|---|---|---|
| figure `cmp_q_legs`, fin de la commande de base (médiane, toutes jambes valides) | 0,813 | 0,864 | 0,510 | 0,632 |
| tableau $R_i$, fin de l'approche (moyenne, runs réussies) | 0,857 | 0,861 | 0,539 | 0,727 |
| même instant, médiane sur toutes les runs valides | 0,858 | 0,859 | 0,532 | 0,726 |

Par ordre d'importance :

1. **L'instant, et c'est l'essentiel.** La courbe de `cmp_q_legs` s'arrête quand la
   *commande de base* s'arrête (verrou d'arrivée, interlock ou convergence). Le bras, lui,
   continue : il reste **20 à 30 s de phase d'approche** après le dernier cycle commandé
   (médiane 22,6 s à la saisie fixe, 22,8 s à la dépose mobile). Le $R_i$ du tableau est
   pris sur les 3 dernières secondes de la phase, donc bien après la fin de la courbe.
   Pendant cet intervalle Q bouge même quand la base ne bouge pas, parce que la carte
   d'occupation change à mesure que le bras traverse la scène.
2. **La source.** La figure trace `ws_mean_0`, ce que le contrôleur calcule à sa propre
   cadence (~1 Hz) ; le tableau utilise `samples.q_goal`, ce que l'enregistreur mesure.
   Même définition de sphère, deux nœuds, deux instants d'évaluation.
3. **La population**, marginale : passer des runs réussies à toutes les runs valides
   déplace le chiffre de 0,004 à la saisie et de 0,007 à la dépose mobile. La seule
   exception est la **dépose fixe** (0,539 → 0,509) : les sept runs qui finissent en
   collision y meurent sur un Q plus bas, et les retirer remonte la moyenne de 0,03.

Piège supplémentaire propre aux courbes alignées dans le temps : la valeur au bord droit
n'est pas « la valeur finale des jambes », c'est la médiane des seules jambes encore en
cours à cet instant. À la dépose mobile la courbe finit à 0,740 alors que la médiane des
dernières valeurs de chaque jambe vaut 0,686.

Piège de colonne, enfin : dans `legs.csv`, **`q_end` n'est pas le $R_i$**. C'est
`q_center`, la moyenne sur les *trois* sphères (but + deux sphères de trajectoire),
c'est-à-dire l'objectif que le contrôleur optimise. La sphère du but seule est
`qgoal_end`. À la saisie du setup 2, les deux racontent des histoires opposées :
`q_end` donne 0,878 fixe contre 0,913 mobile, alors que `qgoal_end` en moyenne donne
0,814 fixe contre 0,800 mobile — la moyenne de `qgoal_end` est tirée vers le bas par six
jambes mobiles finissant entre 0,56 et 0,66, là où la médiane (0,813 contre 0,864) dit
l'inverse. Toute comparaison sur cette colonne doit donner moyenne *et* médiane.

`cmp_q_arrival.png` (et `cmp_q_arrival.csv`) est la figure qui sépare explicitement les
trois instants — début de jambe, fin de commande, fin d'approche — et c'est la bonne
référence pour rapprocher le tableau des courbes.

## Notes de lecture

- **La moyenne de $\lVert\ddot q\rVert$ n'est pas une valeur typique** : la médiane vaut
  0,016 à 0,025 rad/s², soit un sixième de la moyenne. La distribution est un long
  plateau proche de zéro (le bras est immobile ou en régime établi la plupart du temps)
  plus une queue rare. L'écart-type (0,16–0,19) dépasse la moyenne, ce qui rend la
  notation « moyenne ± écart-type » trompeuse ici ; p95 et max disent davantage.
- **La distance base–cible augmente à la saisie du setup 2 mobile** (0,60 → 0,76 m) : le
  contrôleur n'optimise pas la distance, il optimise Q, et à cette étape il recule pour
  dégager le bras de la table. Le $R_i$ final est pourtant identique à celui de la base
  fixe (0,86), donc le recul ne coûte rien.
- Les chiffres d'accélération diffèrent de ceux de la présentation de 0,002–0,003 rad/s²
  (0,106 contre 0,108, etc.) : les maxima sont identiques au centième près, donc c'est
  bien le même signal, mais le lot d'échantillons mis en commun n'est pas exactement le
  même. Les valeurs du script sont celles à publier ; la présentation est à réaligner
  dessus si les deux doivent coexister.
