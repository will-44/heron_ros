# Vitesses — bras et base

Pendant de [`../manipulability/`](../manipulability/README.md) : mêmes
runs, même règle d'exclusion (les 28 cycles écartés n'entrent dans aucun chiffre), même
alignement sur le début de chaque approche.

Reproduire : `python3 analysis/velocity/vel_over_time.py` depuis `heron_experiment/`. Sorties dans `results/analysis/velocity/`.
Sorties : `figures/vel_over_time.png`, `figures/vel_by_phase.png`,
`tables/vel_by_phase.csv`, `tables/vel_legs.csv`, `tables/vel_summary.csv|md`.

Deux grandeurs, toutes deux issues de `joint_states.csv` (~106 Hz) :
`‖q̇‖` (norme du vecteur des six vitesses articulaires) et `|q̇|` moyenné sur les six
axes — ce dernier pour être comparable à la métrique d'accélération de la présentation.

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

## Résumé : l'intuition était la bonne, à une exception près

**Les vitesses instantanées ne distinguent rien.** Le profil est le même dans les quatre
runs : montée à 0,25 rad/s en 3–4 s, puis décroissance. Moyenne de `‖q̇‖` sur l'approche
de saisie 0,073–0,102 rad/s, sur l'approche de dépose 0,079–0,095 rad/s ; p95 autour de
0,23–0,27 rad/s partout. Aucun écart significatif entre base fixe et base mobile sur la
moyenne (p ≥ 0,15 dans les quatre cas). C'est attendu : le MPC borne les vitesses de la
même façon quelle que soit la base, donc cette grandeur mesure surtout le réglage du MPC.

**Ce qui change, c'est l'intégrale, pas le débit.** La course articulaire cumulée
(∫ Σ|q̇| dt, radians parcourus tous axes confondus) sépare nettement les deux conditions :

| Approche de dépose | base fixe | base mobile | p (Mann-Whitney) |
|---|---|---|---|
| course articulaire cumulée — setup 2 | 15,7 rad | **8,5 rad** | 0,000 |
| course articulaire cumulée — setup 1 | 11,3 rad | **8,3 rad** | 0,010 |
| durée de la phase — setup 2 | 89,9 s | 48,9 s | 0,000 |
| max de ‖q̇‖ — setup 2 | 0,527 rad/s | 0,442 rad/s | 0,001 |
| temps quasi immobile (‖q̇‖ < 0,01) — setup 2 | 20,9 s | 14,2 s | 0,003 |

À vitesse égale, le bras fait donc **presque deux fois plus de chemin dans l'espace
articulaire** pour la même tâche quand la base ne se replace pas. Le temps perdu n'est
pas du temps d'attente : c'est du mouvement.

**La signature visible sur les courbes** (`vel_over_time.png`, ligne du bas) : après le
pic initial, l'approche de dépose à base fixe garde une **traîne à 0,02–0,05 rad/s** là où
la médiane porte encore sur la moitié des jambes (jusqu'à ~55 s au setup 2, ~50 s au
setup 1) — le bras continue de ramper sans converger — alors que la base mobile est
retombée sous 0,02 rad/s dès 35 s. Au-delà, la courbe passe en pointillé : quelques jambes
isolées rampent jusqu'à 130 s (setup 1) et 310 s (setup 2), à base fixe comme à base
mobile, mais elles ne décrivent plus le groupe. La vitesse ne dit pas « le bras va plus vite », elle dit
« le bras n'arrête pas de bouger ».

**Sur l'approche de saisie**, les seules différences significatives sont petites et sans
enjeu : p95 de `‖q̇‖` 0,273 contre 0,257 rad/s au setup 1 (−6 %, p < 0,001), course
cumulée 8,81 contre 8,42 rad (p = 0,048), et au setup 2 un temps quasi immobile plus long
à base mobile (10,2 contre 6,8 s, p = 0,017) — le bras attend pendant que la base se
replace. Les courses cumulées de saisie sont équivalentes dans les deux setups : le gain
porte bien sur la dépose.

## Vitesse de la base

Rien à analyser : la commande est strictement en tout-ou-rien. Dans les quatre runs,
`/cmd_vel` ne prend qu'**une seule valeur non nulle, exactement 0,020 m/s**, publiée dans
18–20 % des messages (le contrôleur n'est ouvert que pendant les deux approches). Ni la
saturation adaptative ni le lissage terminal (`taper`) ni la zone morte ne se sont jamais
déclenchés : la loi est bang-bang, comme déjà noté dans les rapports.

## Limites

- **Correction.** Les vitesses ne sont pas celles rapportées par le driver : le logger
  calcule `vel = Δpos/Δt` entre messages `/joint_states`. La différence première résiste
  bien à la gigue d'horodatage (autocorrélation à lag 1 de +0,98, soit un signal et non
  du bruit) et les moyennes et intégrales ci-dessus sont valides, mais les valeurs
  extrêmes ponctuelles (`max de ‖q̇‖`) sont à prendre avec réserve. La double différence,
  elle, ne survit pas — voir [`../acceleration/`](../acceleration/README.md) §0.
- `‖q̇‖` mélange des axes de géométries différentes ; à vitesse cartésienne égale, une
  posture proche d'une singularité produit mécaniquement des vitesses articulaires plus
  élevées. La vitesse du TCP demanderait la jacobienne (cf. la limite notée dans
  l'analyse de manipulabilité) — c'est elle qu'il faudrait regarder pour parler de
  « vitesse du bras » au sens de la tâche.
- La course cumulée est à lire avec la durée de phase à côté : les deux sont corrélées
  par construction, c'est bien le couple (temps, chemin) qui décrit le coût.
