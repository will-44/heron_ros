# Le bras est-il gêné quand la base bouge ?

Deux figures demandées : la vitesse de déplacement de la base contre la qualité Q, et
contre l'accélération maximale du bras.

Reproduire : `python3 base_motion_effect/base_motion_effect.py` depuis `results/`.
Sorties : `figures/base_speed_vs_q.png`, `figures/base_speed_vs_accel.png`,
`figures/base_motion_transitions.png`, `tables/bins.csv`, `tables/summary.csv|md`,
`tables/summary_arm_moving.csv|md`, `tables/matched_arm_speed.csv|md`,
`tables/transitions.csv`.

Même règle d'exclusion que partout ailleurs : les 28 cycles écartés n'entrent dans aucun
chiffre. Unité d'analyse : **une seconde d'approche**, 4 110 secondes sur les deux runs à
base mobile (2 330 base à l'arrêt, 1 616 base en mouvement, 164 entre les deux).
L'accélération est **redérivée** des positions (Savitzky-Golay, fenêtre 0,41 s) : la
colonne `acc_joint_*` du log est du bruit de différentiation, voir
[`../acceleration_analysis/`](../acceleration_analysis/README.md) §0.

## 0. Avertissement de lecture : « la vitesse de la base » n'a que deux valeurs

`hypot(vx, vy)` ne prend jamais que **0 ou 0,020 m/s**. Agrégée par seconde, elle vaut
exactement 0 dans 57 % des secondes d'approche, exactement 0,020 dans 39 %, et tombe entre
les deux dans 4 % (les secondes où la commande s'ouvre ou se ferme). La commande est
bang-bang : ni la saturation adaptative, ni le lissage terminal, ni la zone morte ne se
sont jamais déclenchés.

**Un « Q en fonction de la vitesse » est donc deux traits verticaux, pas une courbe.** Les
figures montrent ce nuage brut tel qu'il est (colonne 1), puis la seule comparaison qui a
du sens — base à l'arrêt contre base en mouvement (colonne 2) — puis un contrôle
(colonne 3).

## 1. Accélération : oui, le bras est plus saccadé pendant que la base roule

| max ‖q̈‖ sur la seconde (rad/s²) | base à l'arrêt | base en mouvement | rapport |
|---|---|---|---|
| setup 1, approche de saisie | 0,089 | **0,800** | ×9,0 |
| setup 1, approche de dépose | 0,134 | **0,807** | ×6,0 |
| setup 2, approche de saisie | 0,085 | **0,668** | ×7,9 |
| setup 2, approche de dépose | 0,097 | **0,731** | ×7,5 |

(p < 10⁻⁴ partout, Mann-Whitney.) Ce rapport brut ne prouve rien : la base est commandée
pendant que le bras fait son grand transit et s'arrête quand le bras se pose. Trois
contrôles, du plus faible au plus fort :

**(a) En ne gardant que les secondes où le bras bouge vraiment** (‖q̇‖ ≥ 0,02 rad/s,
68 % des secondes) l'écart se réduit mais tient : 0,21–0,27 à l'arrêt contre 0,75–0,89 en
mouvement.

**(b) À seconde-de-phase égale** (chaque seconde comparée aux secondes du même instant de
la même phase du même run) : 0,37 → 0,70 et 0,46 → 0,74 au setup 1, 0,26 → 0,55 et
0,39 → 0,54 au setup 2, p entre 0,0005 et 0,04. Peu de secondes s'apparient (7 à 15), donc
c'est indicatif.

**(c) À vitesse de bras égale** — le contrôle décisif, et c'est la colonne 3 de
`base_speed_vs_accel.png`. Si la base ne faisait que coïncider avec les grands mouvements
du bras, les deux nuages se superposeraient. Ils ne se superposent pas :

| ‖q̇‖ médian du décile (rad/s) | 0,029 | 0,057 | 0,094 | 0,14 | 0,19 | 0,23 |
|---|---|---|---|---|---|---|
| rapport mouvement / arrêt — dépose | **×4,2** | ×2,4 | ×2,1 | ×1,6 | ×1,4 | ×1,0 |
| rapport mouvement / arrêt — saisie | **×6,0** | ×3,9 | ×2,1 | ×1,7 | ×1,3 | ×1,7 |

Significatif dans onze déciles sur douze. Dit autrement, le rapport
accélération / vitesse — une mesure d'agitation à vitesse donnée — passe de **3,4 à 5,1**
en dépose et de **3,1 à 4,6** en saisie (+47 %, p < 10⁻⁴). L'effet est le plus fort quand
le bras va lentement, c'est-à-dire précisément quand il n'a aucune raison mécanique
d'accélérer.

**(d) Aux arrêts de la base** (`base_motion_transitions.png`) : sur les 89 arrêts dont on
peut isoler 3 s de part et d'autre, le max de ‖q̈‖ passe de 0,90 à 0,67 (p = 0,0001), en
baisse dans 71 % des cas. À prendre avec précaution : la base s'arrête le plus souvent
parce que le bras est arrivé, donc les deux retombent ensemble. Les démarrages, eux, ne
sont pas mesurables — la base démarre au tout début de la jambe, avant la fenêtre.

**Mécanisme le plus probable.** Le MPC replanifie à chaque cycle contre un but qui se
déplace dans le repère du bras, puisque c'est la base qui bouge sous lui. Cohérent avec ce
qu'on avait déjà mesuré à durée appariée : base mobile = **plus d'inversions
d'accélération mais plus petites** (4,13 contre 3,83 par seconde et par axe, p = 0,017).

## 2. Mais sur la phase entière, la base mobile ne coûte pas plus cher au bras

C'est le contrepoint indispensable, déjà établi dans
[`../acceleration_analysis/`](../acceleration_analysis/README.md) et
[`../velocity_analysis/`](../velocity_analysis/README.md) :

- aucune différence significative de ‖q̈‖ moyenne entre base fixe et base mobile sur les
  quatre comparaisons (p ≥ 0,12) ;
- **à durée appariée, c'est la base fixe qui accélère le plus** (0,182 contre 0,153,
  p = 0,002) ;
- la course articulaire cumulée est presque divisée par deux en dépose (15,7 contre
  8,5 rad, p < 0,001).

Les deux constats se concilient : pendant les ~40 % de secondes où la base roule, le bras
est plus saccadé ; mais la base mobile supprime beaucoup plus de mouvement qu'elle n'en
ajoute, parce que la phase est deux fois plus courte. **Le bras est gêné localement et
soulagé globalement.**

## 3. Qualité Q : la corrélation existe mais se lit à l'envers

| Q sphère du but (médiane) | base à l'arrêt | base en mouvement |
|---|---|---|
| setup 1, approche de saisie | 0,833 | 0,750 |
| setup 1, approche de dépose | 0,773 | 0,745 |
| setup 2, approche de saisie | 0,868 | 0,747 |
| setup 2, approche de dépose | 0,667 | 0,568 |

Q est systématiquement plus bas quand la base bouge (ρ de Spearman −0,20 à −0,49 ; partiel,
à instant de phase contrôlé, −0,00 à −0,48). **Ce n'est pas un effet de la base sur Q,
c'est sa cause :** le contrôleur ne commande la base que lorsque ∇Q est non nul et que
l'interlock l'autorise, c'est-à-dire justement quand Q est encore améliorable ; il
s'arrête quand le bras est arrivé. « Base en mouvement » est donc, par construction, la
sélection des instants où Q est bas.

Le sens réel de l'effet se lit sur le ΔQ pendant la commande, pas sur le niveau : au
setup 2, ΔQ_goal médian sur la jambe de dépose vaut **+0,124 à base mobile contre −0,091 à
base fixe**, positif dans 79 % des jambes contre 12,5 %. Pendant que la base roule, Q
monte.

## Limites

- Pas de mesure mécanique de l'oscillation : pas d'IMU sur le bras, pas de couple
  journalisé. Ce qu'on appelle « saccade » est l'accélération articulaire redérivée des
  positions rapportées, à fenêtre de dérivation fixe (les valeurs absolues dépendent de
  cette fenêtre, pas les rapports).
- Les secondes « base en mouvement » et « base à l'arrêt » ne sont jamais complètement
  comparables : la base bouge tôt dans la jambe et s'arrête sur l'arrivée du bras. Les
  contrôles (a)–(c) réduisent ce biais, ils ne l'annulent pas. Le seul test propre serait
  de commander la base à des instants tirés au hasard dans la phase.
- La base ne tourne jamais (`wz` = 0 dans les 105 137 messages) : tout ceci ne concerne
  que la translation.

## Ce que ça suggère de faire

La commande bang-bang à 0,020 m/s s'ouvre et se ferme d'un coup. Puisque l'effet est
maximal quand le bras va lentement, **mettre une rampe sur `/cmd_vel`** (la saturation
adaptative et le `taper` existent déjà dans `BaseCommander` mais ne se sont jamais
déclenchés, faute d'être atteints) devrait réduire les à-coups sans rien coûter au temps
de tâche, puisque la vitesse de croisière resterait la même.
