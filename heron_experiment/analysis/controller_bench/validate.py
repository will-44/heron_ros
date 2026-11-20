#!/usr/bin/env python3
"""Validation du banc : rejoue hors-ligne un cycle enregistré et le compare
aux Q réellement calculés en ligne par le contrôleur.

Si les Q rejoués coïncident avec `q_scores` du dump, alors le banc reproduit
la chaîne de production et toutes les mesures suivantes sont crédibles.
Aucun node ROS, aucun /cmd_vel.

ATTENTION — les `q_scores` de `control_debug_maps.h5` datent de l'union
booléenne : Q y était la moyenne sur le volume balayé, chaque sphère pesant son
volume *non recouvert*. Le calcul actuel donne à chaque sphère le poids 1/N
(voir WeightedVoxelMask), donc un écart avec ces dumps est attendu dès que la
région compte plus d'une sphère — ce n'est pas une régression du banc. Sur une
région à une seule sphère (cas de ces dumps, capturés avec region_size=1) les
deux calculs coïncident et la validation reste exacte.

ATTENTION (bis) — dans ces dumps, les voxels entrants des 8 candidats translatés
valent 0.0 (le remplissage de l'époque, ni libre ni occupé pour le réseau), et non -1
comme au §1.6 du rapport. Le Package ne prend que de l'occupation : ces voxels y
deviennent libres, donc les cartes rejouées des candidats translatés diffèrent des
cartes enregistrées (jusqu'à 0.8 sur un voxel, 0.08 sur Q). Seul le candidat central,
non translaté, compare le même réseau sur la même entrée : son écart est celui de la
précision de l'engine (0 avec l'engine de l'enregistrement, ~1e-2 entre son fp16 et
PyTorch).
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rm_harness import MAPS_H5, PACKAGE, PRECISION, Harness  # noqa: E402

from heron_experiment.core.reachability_model import ReachabilityModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', default=PACKAGE, help='Heron Package (default: %(default)s)')
    ap.add_argument('--precision', default=PRECISION, help='fp32, fp16 or int8 (default: %(default)s)')
    args = ap.parse_args()
    model = ReachabilityModel(args.package, args.precision)
    print(f'backend = {model.backend}')

    for group in ('1', '2'):
        with h5py.File(MAPS_H5, 'r') as f:
            grp = f['group'][group]
            attrs = dict(grp.attrs)
            if 'q_scores' not in grp:
                continue
            q_rec = np.array(grp['q_scores'])
            obst = np.stack([np.array(grp['candidate'][str(i)]['obstacle_map'])
                             for i in range(9)])
            rm_rec = np.stack([np.array(grp['candidate'][str(i)]['reachability_map'])
                               for i in range(9)])
            masks = np.stack([np.array(grp['candidate'][str(i)]['mask'])
                              for i in range(9)])

        # L'entrée enregistrée est déjà encodée (+1 obstacle, -1 libre) : on en reprend
        # l'occupation, que le Package encode lui-même.
        rm_new = model.predict(torch.as_tensor(obst > 0, device=model.device)).cpu().numpy()

        q_new = np.array([
            float(rm_new[i][masks[i]].mean()) if masks[i].sum() else 0.0
            for i in range(9)])
        q_from_rec = np.array([
            float(rm_rec[i][masks[i]].mean()) if masks[i].sum() else 0.0
            for i in range(9)])

        center = 4  # offset (0, 0) : la scène non translatée
        unknown = int((obst[[i for i in range(9) if i != center]] == 0).sum())
        print(f'\n=== group {group} ===')
        print(f'  candidat central : |RM_rejouée - RM_enregistrée| max = '
              f'{np.abs(rm_new[center] - rm_rec[center]).max():.2e}')
        print(f'  candidats translatés : {unknown} voxels entrants à 0.0 dans l\'entrée '
              f'enregistrée (libres ici), écarts attendus')
        print(f'  Q enregistrés   : {np.round(q_rec, 5)}')
        print(f'  Q recalculés (RM enregistrée) : {np.round(q_from_rec, 5)}')
        print(f'  Q rejoués    (RM réinférée)   : {np.round(q_new, 5)}')
        print(f'  |Q_rejoué - Q_enregistré| max = {np.abs(q_new - q_rec).max():.2e}')
        print(f'  |RM_rejouée - RM_enregistrée| max = '
              f'{np.abs(rm_new - rm_rec).max():.2e}')

        gx, gy, _, res = Harness.fit_plane(q_rec)
        gxr, gyr, _, _ = Harness.fit_plane(q_new)
        print(f'  grad enregistré  : ({attrs["grad_x"]:+.5f}, {attrs["grad_y"]:+.5f}) '
              f'|{attrs["grad_mag"]:.5f}|')
        print(f'  grad refité (Q enr.): ({gx:+.5f}, {gy:+.5f}) '
              f'|{np.hypot(gx, gy):.5f}|  residu={res:.5f}')
        print(f'  grad rejoué      : ({gxr:+.5f}, {gyr:+.5f}) '
              f'|{np.hypot(gxr, gyr):.5f}|')


if __name__ == '__main__':
    main()
