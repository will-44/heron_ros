# Ce que fait réellement le contrôleur de base HERON

> **Depuis la migration vers heron_ros** : les chemins `capacitynet/...` et les numéros de
> ligne cités ci-dessous sont ceux du code audité, avant son passage dans
> `heron_experiment/heron_experiment/{core,components,nodes}/` (voir `docs/ARCHITECTURE.md`).
> Les mesures ont été faites avec `/home/ros2_ws/src/test.yaml` et l'engine
> `config/unet3d.trt` ; le banc charge maintenant le Heron Package déployé (`rm_harness.py`).
> Avec PyTorch, il redonne les Q de l'ancien banc à 1,2e-7 près ; `validate.py` explique
> pourquoi les candidats translatés du dump ne se comparent plus voxel à voxel.

Audit statique (phase 1) et mesures passives (phase 2). Robot immobile pendant
toute l'étude ; aucun fichier du contrôleur de production n'a été modifié.

**Code audité** : `capacitynet/capacitynet/control/{scripts,components,nodes}/`
**Config déployée** : `capacitynet/config/gradient_base_controller_params.yaml` + `/home/ros2_ws/src/test.yaml`
**Banc de mesure** : `rm_harness.py`, validé bit-à-bit contre la production
(RM rejouée = RM enregistrée, écart max `0.00e+00` ; gradient reproduit à l'identique
sur deux cycles réels).

---

## 1. Tableau hypothèse / statut / preuve

### Phase 1 — audit statique

| # | Hypothèse | Statut | Preuve |
|---|---|---|---|
| 1.1a | Le modèle local est un **plan** (3 coefficients), pas une quadratique | **CONFIRMÉ** | `gradient_controller.py:325` — `grad_x = Σ(xᵢQᵢ)/Σxᵢ²`, aucun terme d'ordre 2 |
| 1.1b | Aucune hessienne / courbure n'est calculée dans la boucle de commande | **CONFIRMÉ** | Aucune occurrence dans `gradient_controller.py` ni `base_commander.py`. Le seul ajustement quadratique du paquet est `control_debug_logger.py:341-360`, réservé au HTML Plotly de debug, sur des points accumulés *entre* cycles, jamais utilisé pour commander |
| 1.1c | Le résidu de l'ajustement est calculé ou journalisé | **INFIRMÉ** | `evaluate()` (`gradient_controller.py:368-404`) ne renvoie pas de résidu ; `_FIELDS` du CSV (`control_debug_logger.py:76-83`) n'a pas de colonne résidu |
| 1.2a | Le pas est **normalisé** (`r·g/‖g‖`) | **INFIRMÉ** | `base_commander.py:60-68` — `vx = gain·grad_x`, puis mise à l'échelle **seulement si** `magnitude > max_vel` : loi proportionnelle à saturation |
| 1.2b | Le pas effectif vaut `r = v_max × T` | **CONFIRMÉ, r = 0.02 m** | `base_speed: 0.02` m/s (cfg L95) × `cycle_period: 1.0` s (cfg L29) |
| 1.2c | Le comportement observé est néanmoins celui d'un pas fixe | **CONFIRMÉ par la mesure** | 88.5 % des cycles réels en saturation (logs, phase 1) ; 98.3 % des voisinages de la carte dense (phase 2, §2.6) |
| 1.3a | Il existe un seuil sur ‖g‖ sous lequel la commande est nulle | **CONFIRMÉ** | `base_commander.py:272` — `if speed < self.min_speed`, avec `min_speed: 0.005` m/s (cfg L101) ⇒ seuil `‖g‖ < 0.025` 1/m |
| 1.3b | Ce seuil est une simple zone morte | **INFIRMÉ — c'est un verrou** | `base_commander.py:273` pose `self._converged = True`, état **latché** ; seule sortie : `_set_enabled` (`base_commander.py:357`). Le contrôleur ne redémarre pas tout seul |
| 1.3c | Il existe un filtre / lissage temporel sur g ou sur la commande | **INFIRMÉ** | Aucun filtre sur g. L'unique EWMA (`base_commander.py:254`) porte sur le **temps de cycle**, pas sur le gradient. Le taper quadratique (`base_commander.py:266`) est désactivé en déploiement (`gradient_taper_ref: 0.0`, cfg L107) |
| 1.3d | Un ajustement aberrant est détecté et traité | **INFIRMÉ** | Aucune détection. `compute_mean` renvoie `0.0` pour un masque vide (`voxel_mask.py:219`), ce qui produit un Q artificiel entrant tel quel dans l'ajustement |
| 1.4a | `delta = 0.1 m` | **CONFIRMÉ** | `grid_spacing: 0.1` (cfg L87) |
| 1.4b | `Σxᵢ = Σyᵢ = Σxᵢyᵢ = 0` et `Σxᵢ² = 6δ²` | **CONFIRMÉ numériquement** | `grid_offsets()` (`gradient_controller.py:202-218`) ⇒ x = (−δ,0,+δ)×3 : Σx = 0, Σy = 0, Σxy = 0, Σx² = 6δ² = 0.06. L'estimateur `Σ(xᵢQᵢ)/Σxᵢ²` est donc **exactement** la solution des moindres carrés, non biaisée |
| 1.4c | `delta / voxel_size` est entier | **CONFIRMÉ** | `voxel_size = 0.02` m (attribut HDF5) ⇒ 0.1/0.02 = **5**, entier. `mask_shift_exact = True` (`gradient_controller.py:272`), chemin par translation entière (`:286`) |
| 1.5 | `gradient_method` déployé = `least_squares` (les 9 échantillons utilisés) | **CONFIRMÉ** | cfg L111 `"least_squares"` ; défaut code identique (`reachability_pipeline.py:87`). Le mode `central` (4 échantillons sur 9, `gradient_controller.py:319`) n'est **pas** actif |
| 1.6a | 9 inférences réseau distinctes sur 9 grilles translatées | **CONFIRMÉ** | `reachability_pipeline.py:229-233` : `generate_grid_transforms` empile 9 volumes, puis **un seul** `predict_batch` de lot 9 — 9 inférences, un appel |
| 1.6b | Décalage entier de voxels | **CONFIRMÉ** | `obstacle_transformer.py:127-134` — `di = -int(round(delta_x/resolution))` |
| 1.6c | Les voxels entrants sont marqués « inconnu » | **INFIRMÉ** | `obstacle_transformer.py:134` remplit à `-1`. Le commentaire annonce « inconnu, distinct de 0=libre », mais l'entrée est **déjà normalisée** par `preprocess` (obstacle→+1, libre→−1) : le réseau lit ces voxels comme de l'**espace libre**, pas comme de l'inconnu |
| 1.7a | Φ est une moyenne sur **plusieurs sphères** | **CONFIRMÉ** | `voxel_mask.py:423` `union_of_spheres` ; 1 sphère de goal + 1 par entrée de `path_horizon_s` |
| 1.7b | Pondération moyenne-de-moyennes ⇒ goal 1/3, chemin 2/3 | **INFIRMÉ dans le mécanisme** | C'est une **union** booléenne : le recouvrement n'est compté qu'une fois (`voxel_mask.py:423-468`, dont le commentaire note que des poids par région « exigeraient un masque flottant »). La pondération est donc **proportionnelle au volume** et dérive quand les sphères se recouvrent. Avec 3 rayons égaux et sphères disjointes on retrouve 1/3 par coïncidence, pas par construction |
| 1.7c | Horizons de chemin à 1 s et 2 s | **INFIRMÉ** | Déployé : `path_horizon_s: "4.0,1.0"` (cfg L48). Le défaut du code est `"2.0,1.0"` (`workspace_region.py:84`) — ni l'un ni l'autre ne correspond à l'hypothèse. Paramétrable, pas codé en dur |
| 1.7d | Rayons | **CONFIRMÉ** | `workspace_radius: 0.10` (cfg L80) pour le goal, `path_radius: 0.10` (cfg L84) pour les points de chemin. Défauts code : 0.10 et **0.05** (`workspace_region.py:90,95`) |
| 1.8a | Période de boucle = `cycle_period` | **CONFIRMÉ, 1.0 s** | cfg L29 ; timer (`reachability_pipeline.py:163`) découplé de l'abonnement, qui ne fait que cacher (`:189`) |
| 1.8b | Les trois latences 100 / 450 / 600 ms | **RÉSOLUES, aucune n'est la latence bout-en-bout** | ~100 ms : ancienne mesure PyTorch par carte. **~440 ms** : GPU total du cycle à 9 candidats (`OPTIMIZATIONS.md:48`), cohérent avec la table TRT 39.8 ms (lot 1) → 369.8 ms (lot 9), strictement linéaire. **608 ms** : historique, TRT FP16 lot 1 à 152³ avec les anciens `f_maps` (`OPTIMIZATIONS.md:195`), plus applicable |
| 1.8c | Latence perception → commande | **MESURÉE : ~1.2 s en moyenne** | Péremption moyenne de la grille cachée (~0.5 s, demi-période du timer) + cycle complet mesuré `cycle_s` médiane **0.688 s** (p95 0.793 s, logs) + publication (`publish_rate: 10.0` Hz, cfg L120). Aucune des trois valeurs 100/450/600 ms n'approche ce total |
| 1.8d | La commande vieillit au-delà du cycle quand des cycles sont sautés | **CONFIRMÉ** | Bloqueur d'ordre zéro : `dt_since_last_s` médiane 1.01 s mais **p95 4.92 s**, max 11.0 s (logs). La commande appliquée peut donc avoir près de 5 s d'âge |

### Phase 2 — mesures passives

| # | Hypothèse | Statut | Chiffre mesuré |
|---|---|---|---|
| 2.1a | σ de prédiction est mesurable et non nul | **INFIRMÉ** | σ = **0.000e+00** sur 120 tirages, aux trois distances (Q = 0.868, 0.426, 0.757) : **1 seule valeur distincte sur 120** à chaque fois. TensorRT est déterministe au bit près |
| 2.1b | σ dépend de la valeur de Q | **SANS OBJET** | σ = 0 quel que soit Q |
| 2.1c | Le chemin CUDA Graph (lot 1) et le chemin `execute_async_v3` (lot ≥ 2) donnent le même Q | **CONFIRMÉ** | écart `+0.000e+00` aux trois distances |
| 2.2a | Le résidu intra-cycle dépasse le σ temporel | **CONFIRMÉ trivialement** | résidu plan médian **0.0169** (p95 0.081) contre σ = 0 : le résidu est **100 % courbure réelle**, 0 % bruit |
| 2.2b | Un résidu élevé dégrade le gradient estimé | **INFIRMÉ** | Le résidu des moindres carrés est orthogonal au design par construction : `Σxᵢrᵢ = −7.4e−20`, `Σyᵢrᵢ = −2.2e−17`. Il ne se propage **pas** dans (gx, gy) |
| 2.2c | Les erreurs entre les 9 poses sont indépendantes | **INFIRMÉ** | Corrélations hors-diagonale de **−0.86 à +0.89** (moyenne −0.107), structure déterministe. Biais par pose : centre **+0.0156** au-dessus du plan, coins ±x∓y à −0.024 / −0.023 |
| 2.3a | Une part notable du masque est faite de voxels entrants | **INFIRMÉ** | **0.000 %** du masque pour les 8 poses. (3.906 % du volume 128³ pour les poses axiales, 7.660 % pour les diagonales) |
| 2.3b | Le biais des voxels entrants est asymétrique | **CONFIRMÉ mais petit** | contenu observé perdu : **+109** voxels occupés entre +x et −x, **−234** entre +y et −y, sur ~1.8e5 voxels occupés |
| 2.4a | Le champ Q est unimodal sur ±0.2–0.4 m | **INFIRMÉ** | **21 maxima locaux stricts** sur la carte 41×41 à 2 cm |
| 2.4b | Le champ est lisse (pas de discontinuité) | **NUANCÉ** | Pas de falaise, mais saut max de **0.0797** entre deux cases adjacentes de 2 cm ; Q ∈ [0.104, 0.941] |
| 2.4c | L'optimum est dans la fenêtre | **INFIRMÉ** | Maximum en (−0.30, +0.36) m, **sur le bord** de la fenêtre ±0.40 m |
| 2.5a | H est définie négative près de l'optimum | **INFIRMÉ** | Définie négative dans **30.5 %** seulement des 961 voisinages. λ₁ médian −3.47, λ₂ médian **+0.90** : la surface est majoritairement un **point-selle** |
| 2.5b | Rapport d'anisotropie | **MESURÉ** | médiane **3.68**, p95 39.5, moyenne 25.4 |
| 2.5c | L'angle entre g et la direction de Newton est < 5° (⇒ Newton inutile) | **INFIRMÉ** | médiane **44.3°** ; > 5° dans **92.8 %** des cas. (Au centre exact de la scène testée : 3.5°, cas favorable minoritaire) |
| 2.5d | Passer à un modèle quadratique améliorerait la **direction** | **INFIRMÉ** | Sur un réseau 3×3 symétrique, `max ‖g_plan − g_quadratique‖ = 3.55e−15` sur 961 voisinages. Les colonnes x, y sont orthogonales à 1, x², y², xy : la courbure ne change **pas** (gx, gy), elle ne fournit que H |
| 2.6 | Le régime est celui d'une ascension à pas fixe, pas d'un Newton tronqué | **CONFIRMÉ** | ‖H⁻¹g‖ médian **0.179 m** contre r = 0.02 m ; **99.9 %** des voisinages dépassent r. Régime commande : **98.3 % saturé**, 1.7 % proportionnel, 0 % zone morte |
| 2.7a | Distribution de ‖∇Q‖ | **MESURÉE** | Carte dense : médiane **0.382** 1/m (p05 0.130, p95 1.817). Logs réels : médiane 0.237, p05 0.079, p95 2.51 — même ordre |
| 2.7b | Le bruit fixe le seuil de fiabilité de la direction | **INFIRMÉ** | `σ_g = σ/(h√6) = 0` puisque σ = 0. Le bruit du modèle ne dégrade **jamais** la direction ; la zone morte ne peut pas être dimensionnée à partir de σ |
| 2.7c | La direction estimée à δ=0.1 m égale le gradient local réel | **INFIRMÉ** | Angle avec le gradient à 2 cm : médiane **37.5°**, > 20° dans 70.4 %, **> 90° dans 19.6 %**. Rapport d'amplitude médian 1.03. L'écart vient de l'**échelle d'échantillonnage**, pas du bruit |

---

## 2. Description exacte du contrôleur, en une équation

À chaque cycle de période fixe `T = 1.0 s`, sur la dernière grille d'occupation
reçue (mise en cache, non synchronisée avec le cycle) :

Neuf cartes d'obstacles sont produites par translation entière de la grille
courante aux offsets `δᵢ = (xᵢ, yᵢ) ∈ {−δ, 0, +δ}²`, `δ = 0.10 m` (= 5 voxels) ;
les voxels entrants sont remplis à la valeur « libre ». Le réseau les évalue en
un lot de 9. Chaque carte de reachability est réduite à un scalaire par moyenne
sur une **union de sphères** (goal, rayon 0.10 m, plus une sphère de rayon 0.10 m
par horizon de `path_horizon_s = {4.0 s, 1.0 s}` du chemin MPC), le masque étant
translaté du même nombre entier de voxels :

```
    Qᵢ  =  (1/|M ⊕ δᵢ|) · Σ_{v ∈ M ⊕ δᵢ}  RM_i(v)
```

Le gradient est l'ajustement **par moindres carrés d'un plan** (3 coefficients)
sur ces 9 points, qui se réduit en forme fermée sur ce réseau symétrique à :

```
    ĝ  =  ( Σᵢ xᵢQᵢ / Σᵢ xᵢ² ,  Σᵢ yᵢQᵢ / Σᵢ yᵢ² ),      Σᵢ xᵢ² = Σᵢ yᵢ² = 6δ²
```

La commande de vitesse est **proportionnelle avec saturation**, suivie d'une
**zone morte verrouillante** :

```
                ⎧  0                                    si ‖k·ĝ‖ < v_min      (état CONVERGÉ, latché)
    v(t)   =    ⎨  k · ĝ                                si v_min ≤ ‖k·ĝ‖ ≤ v_max
                ⎩  v_max · ĝ/‖ĝ‖                        si ‖k·ĝ‖ > v_max

    k = 0.2 m²/s,   v_max = 0.02 m/s,   v_min = 0.005 m/s
```

`v` est tenu constant (bloqueur d'ordre zéro) sur `/cmd_vel` à 10 Hz jusqu'au
cycle suivant, avec chien de garde à 2.10 s.

**Formulation pour l'article.** Puisque 88.5 % des cycles réels (et 98.3 % des
voisinages échantillonnés) tombent dans la branche saturée, le contrôleur est,
en régime nominal, une **ascension de gradient à pas fixe** :

```
    p_{t+1}  =  p_t  +  r · ĝ / ‖ĝ‖ ,      r = v_max·T = 0.02 m
```

Il faut cependant écrire que la normalisation est **obtenue par saturation** et
non imposée : le code implémente `v = clip(k·ĝ, v_max)`, ce qui n'est pas la même
chose qu'une normalisation explicite dans le régime de faible gradient (branche
proportionnelle) ni près du seuil `v_min`. Décrire le contrôleur comme
« normalisé » serait inexact ; « proportionnel saturé, saturé 88.5 % du temps,
donc effectivement à pas fixe » est la formulation défendable.

**Modèle local** : linéaire (plan, 3 coefficients). **Pas** : proportionnel
saturé. **Zone morte** : oui, `v_min = 0.005 m/s` soit `‖ĝ‖ < 0.025` 1/m, et
elle est **verrouillante**.

---

## 3. Ce que le code fait et qui n'était pas soupçonné

Par ordre décroissant d'importance pour l'article.

1. **La zone morte est un verrou, pas un seuil.** Franchir `min_speed` une seule
   fois pose `_converged = True` (`base_commander.py:273`), et cet état ne se
   lève que par un appel explicite à `~/enable` (`base_commander.py:357`). Le
   contrôleur ne repart jamais de lui-même, même si le gradient redevient grand.
   Toute description en « le robot ralentit puis s'arrête près de l'optimum »
   est fausse : il s'arrête **définitivement**, à la première violation.

2. **Le pipeline n'infère que sur 0.64 % des tops de timer.** Sur les 16 203
   lignes de `results/run_*/control_debug.csv`, **104** correspondent à une
   inférence ; 16 096 sortent en `gate_closed` et 3 en `region_out_of_grid`. Le
   contrôleur passe l'écrasante majorité de son temps à ne rien calculer : la
   « période de boucle » de 1.0 s n'est pas la période effective d'actualisation
   de la commande, qui vaut `dt_since_last_s` — médiane 1.01 s, **p95 4.92 s**,
   max 11.0 s. Ne pas citer 1 Hz comme fréquence de commande dans l'article.

3. **Il existe une saturation adaptative, jamais active en déploiement.**
   `_velocity_cap()` (`base_commander.py:329-334`) calcule
   `cap = step_fraction·δ/EWMA(cycle_s)` = `0.5·0.1/cycle_s`, soit 0.05 m/s pour
   un cycle de 1 s, puis prend `min(base_speed, cap)`. Comme `base_speed = 0.02`,
   la borne adaptative ne mord jamais tant que le cycle reste sous 2.5 s. Le
   mécanisme existe, il est inerte.

4. **Le facteur d'atténuation quadratique existe et est désactivé.**
   `gradient_taper_ref: 0.0` (cfg L107) court-circuite le bloc
   `base_commander.py:266-269`. Ne pas décrire le contrôleur comme ayant une
   décélération douce à l'approche : il n'en a pas.

5. **Les voxels entrants sont vus comme de l'espace libre, pas comme de
   l'inconnu.** Le remplissage `-1` (`obstacle_transformer.py:134`) est appliqué
   *après* la normalisation en [−1, +1], où −1 signifie « libre ». Le
   commentaire du code affirme le contraire. Conséquence physique : chaque pose
   candidate décalée « invente » 3.9 % (axiale) à 7.7 % (diagonale) de volume
   libre au bord. Mesuré sans effet dans le masque (0.000 %), mais l'effet via
   le champ récepteur du réseau reste non quantifié.

6. **Ajouter un modèle quadratique ne changerait pas la direction commandée.**
   Résultat exact, pas empirique : sur un réseau 3×3 symétrique les colonnes
   linéaires du design sont orthogonales aux colonnes quadratiques, donc le
   gradient de l'ajustement à 6 coefficients est **identique** à celui du plan
   (écart 3.55e−15 sur 961 voisinages). L'argument « on n'utilise qu'un plan,
   donc on perd de l'information directionnelle » est faux.

7. **Le résidu ne mesure pas une erreur de gradient.** Orthogonal au design par
   construction (`Σxᵢrᵢ ≈ 1e−19`). Le journaliser renseignerait sur la validité
   du modèle local, pas sur la qualité de ĝ.

8. **La surface Q est majoritairement un point-selle, pas un dôme.** λ₂ > 0 dans
   69.5 % des voisinages. C'est l'argument le plus fort contre une méthode de
   second ordre : −H⁻¹g y pointe vers le col.

9. **Q n'est pas une moyenne de moyennes.** L'union booléenne
   (`voxel_mask.py:423`) compte le recouvrement une seule fois, si bien que le
   poids de chaque sphère est proportionnel à son volume **non recouvert** et
   varie donc au cours de la trajectoire, à mesure que le goal et la queue du
   chemin MPC se rapprochent. Le « 1/3 goal, 2/3 chemin » n'est vrai que dans le
   cas disjoint.

10. **Le point d'horizon se sature silencieusement.** `_path_horizon_center_at`
    (`workspace_region.py:485`) prend `min(..., key=|stamp − (now + h)|)` : si
    l'horizon dépasse la durée du chemin MPC, il renvoie la **dernière** pose
    sans le signaler. Avec `path_horizon_s = "4.0,1.0"` et un chemin MPC plus
    court que 4 s, les sphères « 4 s » et « goal » se confondent, et la
    pondération effective bascule.

11. **Un masque vide produit Q = 0.0, pas une erreur.** `voxel_mask.py:219`.
    Ce 0.0 entre tel quel dans l'ajustement et fabrique un gradient énorme
    pointant à l'opposé de la pose concernée. Aucune détection en aval.

12. **Le réseau est parfaitement déterministe.** Répétabilité au bit près sur
    120 tirages, y compris entre le chemin CUDA Graph (lot 1) et le chemin
    standard (lot 9). Toute variabilité de Q observée en ligne vient de la
    perception ou de la TF, jamais du modèle.

13. **L'estimateur du gradient est exactement non biaisé.** `Σxᵢ = Σyᵢ = Σxᵢyᵢ = 0`
    et `Σxᵢ² = 6δ² = 0.06` : la forme fermée `Σ(xᵢQᵢ)/Σxᵢ²` coïncide avec la
    solution des moindres carrés. Rien à corriger de ce côté.

14. **Un seul appel réseau, pas neuf.** `predict_batch` de lot 9
    (`reachability_pipeline.py:233`) — 9 inférences, un lancement. Le coût
    mesuré (369.8 ms pour le lot 9 vs 39.8 ms pour le lot 1) est strictement
    linéaire : le lot n'apporte aucun gain d'occupation GPU sur cette
    architecture.

---

## 4. Écarts entre le code, la configuration et la documentation

| Écart | Détail |
|---|---|
| **`CLAUDE.md` décrit une arborescence obsolète** | Il annonce `capacitynet/capacitynet/gradient_controller.py`, `workspace_evaluation.py`, etc. à la racine du paquet. La réalité est `capacitynet/capacitynet/control/{components,nodes,scripts}/` et `capacitynet/capacitynet/rm_model/`. Aucun des chemins cités n'existe |
| **Défauts du code ≠ config déployée, sur 5 paramètres critiques** | `cycle_period` 2.0 → **1.0** ; `path_horizon_s` "2.0,1.0" → **"4.0,1.0"** ; `path_radius` 0.05 → **0.10** ; `workspace_radius` code du `GradientBasedController` 0.30 → **0.10** ; `control_gain` 0.01 → **0.2** (facteur 20). Un lancement sans le YAML donne un contrôleur au comportement radicalement différent |
| **Commentaire faux sur le remplissage** | `obstacle_transformer.py:126-134` affirme que `-1` code l'inconnu « distinct de 0=libre ». Après `preprocess`, −1 **est** la valeur du libre |
| **`OPTIMIZATIONS.md` contient deux générations de chiffres** | L20 donne 128³ (à jour) mais le texte parle ailleurs de 152³ ; L195 cite 608 ms pour un lot 1, valeur d'une configuration abandonnée. La seule table valide est L32-36 |
| **`dsr_msgs2` utilisé mais non déclaré** | `grasp.py` en dépend, absent de `package.xml`. Hors périmètre de cette étude, signalé au passage |
| **Les dumps HDF5 ne reflètent plus la config** | `control_debug_maps.h5` a été capturé avec `region_size=1`, `region_source='marker'`, rayon ≈ 0.30 m (le défaut code). Le déploiement utilise 3 sphères à 0.10 m. Les mesures de la phase 2 ont été refaites avec la géométrie déployée, mais sur la **scène** de ces dumps |

---

## 5. Recommandations, classées par bénéfice / risque

*Aucune de ces pistes n'est appliquée. Aucune modification du contrôleur de
production n'est proposée ici pour exécution — la décision appartient à
l'auteur.*

### A. Coût nul, risque nul — observabilité seulement

- **Journaliser le résidu du plan** (`res_lin`) et le nombre de voxels de chaque
  masque dans `control_debug.csv`. Purement additif, aucune influence sur la
  commande. Donnerait, sur données réelles, la statistique de validité du modèle
  local que la phase 2 n'a pu établir que sur une scène.
- **Journaliser `skip_reason` agrégé** pour établir la fréquence effective de
  commande (point 3.2), chiffre nécessaire à l'article.
- **Corriger le commentaire** de `obstacle_transformer.py:126-134` et
  l'arborescence de `CLAUDE.md`. Documentation seule.

### B. À évaluer — demanderait une campagne dédiée avant toute décision

- **Le verrou de convergence** (point 3.1). C'est le comportement le plus
  susceptible d'être décrit à tort dans l'article. Mesurer d'abord sa fréquence
  de déclenchement sur les logs existants.
- **La saturation du point d'horizon** (point 3.10) : instrumenter
  `_path_horizon_center_at` pour compter les cas où l'horizon dépasse le chemin.
- **Le masque vide → Q = 0** (point 3.11) : vérifier sur logs si le cas se
  produit jamais en pratique avant de considérer qu'il faut le traiter.

### C. À ne pas entreprendre — la mesure le déconseille

- **Estimer la hessienne.** Trois raisons cumulées, chacune suffisante :
  H n'est définie négative que dans 30.5 % des voisinages (Newton pointerait
  vers le col ailleurs) ; ‖H⁻¹g‖ dépasse r dans 99.9 % des cas (le pas serait de
  toute façon tronqué à 2 cm, seule la direction compterait) ; et le passage au
  modèle quadratique ne change pas ĝ (écart 3.55e−15). L'angle médian de 44.3°
  ne plaide donc pas pour Newton — il révèle surtout que la surface n'est pas
  convexe localement.
- **Toucher à `grid_offsets` / à la forme fermée du gradient.** L'estimateur est
  exactement non biaisé (point 3.13). Il n'y a rien à y gagner.
- **Réduire `δ` pour « mieux » suivre le gradient.** L'écart de 37.5° entre
  l'estimation à δ=0.1 m et le gradient à 2 cm ne prouve pas une erreur : sur un
  champ portant 21 maxima locaux, δ=0.1 m agit comme un lissage probablement
  souhaitable pour un pas de 2 cm. Trancher exigerait une comparaison en boucle
  fermée, impossible robot immobile.

---

## 6. Chiffres réutilisables dans l'article

| Grandeur | Valeur | Source |
|---|---|---|
| Bruit de prédiction σ (entrée fixe) | **0** (déterministe au bit près, 120 tirages × 3 poses) | mesuré, §2.1 |
| Résidu de l'ajustement plan (médiane / p95) | **0.0169** / 0.0810 | mesuré, 961 voisinages |
| Résidu de l'ajustement quadratique (médiane) | 0.0092 (rapport plan/quad = 1.81) | mesuré |
| ‖∇Q‖ (médiane, p05, p95) — carte dense | **0.382**, 0.130, 1.817 1/m | mesuré |
| ‖∇Q‖ (médiane, p05, p95) — logs réels | 0.237, 0.079, 2.51 1/m | logs `control_debug.csv` |
| Valeurs propres de H (médianes) | λ₁ = **−3.47**, λ₂ = **+0.90** | mesuré |
| H définie négative | **30.5 %** des voisinages | mesuré |
| Rapport d'anisotropie \|λmax\|/\|λmin\| (médiane / p95) | **3.68** / 39.5 | mesuré |
| Angle(ĝ, −H⁻¹ĝ) (médiane) | **44.3°** ; > 5° dans 92.8 % | mesuré |
| ‖H⁻¹ĝ‖ (médiane) vs r = 0.02 m | **0.179 m** ; > r dans **99.9 %** | mesuré |
| Fraction de cycles en saturation | **88.5 %** (logs réels) / 98.3 % (carte dense) | logs + mesuré |
| Angle entre ĝ (δ=0.1 m) et ∇Q local (2 cm) | médiane **37.5°** ; > 90° dans 19.6 % | mesuré |
| Voxels entrants — volume 128³ | 3.906 % (axial) / 7.660 % (diagonal) | mesuré |
| Voxels entrants — **dans le masque** | **0.000 %** | mesuré |
| Maxima locaux sur ±0.40 m à 2 cm | **21** | mesuré |
| Latence d'inférence, lot 1 / lot 9 | **39.8 ms** / **369.8 ms** (linéaire) | `OPTIMIZATIONS.md:32-36`, recoupé à 42 ms/carte au banc |
| GPU total par cycle | ~440 ms | `OPTIMIZATIONS.md:48` |
| Cycle complet mesuré `cycle_s` (médiane / p95) | **0.688 s** / 0.793 s | logs `control_debug.csv` |
| Période effective de commande `dt_since_last_s` (médiane / p95 / max) | **1.01 s** / **4.92 s** / 11.0 s | logs |
| Fraction de tops de timer donnant une inférence | **0.64 %** (104 / 16 203) | logs |
| **Latence bout-en-bout perception → commande** | **~1.2 s en moyenne** (jusqu'à ~5 s quand des cycles sont sautés) | dérivée, §1.8c-d |
| Pas commandé r = v_max·T | **0.02 m** | cfg L95 × L29 |
| Seuil de zone morte en gradient | **‖ĝ‖ < 0.025 1/m** | cfg L101 / L104 |
| Pas d'échantillonnage δ | **0.10 m = 5 voxels** (voxel 0.02 m) | cfg L87 |

---

## Annexe — reproductibilité

```
rm_harness.py    banc passif (aucun rclpy, aucun /cmd_vel)
validate.py      validation bit-à-bit contre la production
measure.py       mesures 2.1, 2.3, 2.4, 2.2/2.5/2.6/2.7
figures.py       matrice de corrélation + figures
addendum.py      orthogonalité du résidu, plan vs quadratique, erreur de direction
phase1_logstats.py  statistiques sur results/run_*/control_debug.csv
results/controller_bench/20260811_152423_debug_maps/{tables,figures}/             report.json, dense_q.npz, neighbourhoods.npz, 3 figures PNG
```

Scène : `group 0` de `results/controller_bench/20260811_152423_debug_maps/raw/control_debug_maps.h5`, 128³, voxel
0.02 m, origine −1.28 m, frame `dsr01/world`. Modèle : `/home/ros2_ws/src/test.yaml`
→ `config/unet3d.trt` (TensorRT, `max_batch = 10`).

**Limite principale** : toutes les distributions de la phase 2 proviennent d'une
**seule scène statique** et d'une **sphère de goal unique** (r = 0.10 m). Elles
décrivent ce voisinage, pas une statistique sur l'ensemble des scènes
rencontrées. Les répliquer sur plusieurs cycles enregistrés demanderait de
nouvelles captures via `~/save_reachability_maps`.
