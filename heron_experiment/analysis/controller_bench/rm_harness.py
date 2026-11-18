#!/usr/bin/env python3
"""Banc de mesure passif du contrôleur de base — AUCUN MOUVEMENT POSSIBLE.

Ce module ne crée aucun node ROS, n'importe pas rclpy, et ne construit donc
aucun publisher /cmd_vel. Toutes les poses candidates sont virtuelles : elles
sont obtenues en translatant la carte d'obstacles, exactement comme le fait
`ObstacleMapTransformer` en production. Le robot ne peut pas bouger.

Scène : un cycle réel capturé par `~/save_reachability_maps`, rangé dans
`results/controller_bench/20260811_152423_debug_maps/raw/control_debug_maps.h5`
(frame dsr01/world, 128³, voxel 0.02 m, origine -1.28 m). La carte d'obstacles du
candidat central (offset 0,0) est la scène non translatée telle qu'elle a été
donnée au réseau.

Modèle : le Heron Package réellement déployé (config/gradient_base_controller_params.yaml :
`heron_package`, `precision`), chargé par heron.inference, avec repli PyTorch si son
engine n'est pas construit sur cette machine. Le Package encode lui-même l'entrée : le
banc ne manipule que de l'occupation, comme le contrôleur.

Reproduit à l'identique la chaîne de production :
    translate_filled(occupation, fill=False)  ->  ReachabilityModel.predict  ->
    masque de sphères translaté du même nombre entier de voxels  ->  moyenne

Lancer depuis heron_experiment/ : python3 analysis/controller_bench/measure.py, puis
figures.py et addendum.py. Les sorties vont dans tables/ et figures/ du run du dump.
"""

import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runs  # noqa: E402

from heron_experiment.core.reachability_model import ReachabilityModel  # noqa: E402
from heron_experiment.core.voxel_mask import translate_filled  # noqa: E402
from heron_experiment.core.workspace_evaluation import WorkspaceEvaluation  # noqa: E402

RUN = 'controller_bench/20260811_152423_debug_maps'
MAPS_H5 = str(runs.raw(RUN) / 'control_debug_maps.h5')
TABLES = str(runs.run_dir(RUN) / 'tables')     # report.json, dense_q.npz, neighbourhoods.npz
FIGURES = str(runs.run_dir(RUN) / 'figures')
PACKAGE = 'full_setup_20260805_1144'
PRECISION = 'fp16'
CANDIDATES = 9      # Scenes per predict() call: the controller's 3x3 grid

# Config réellement déployée (config/gradient_base_controller_params.yaml)
DELTA = 0.1
GRID_SIZE = 3
WORKSPACE_RADIUS = 0.10
PATH_RADIUS = 0.10
CONTROL_GAIN = 0.2
BASE_SPEED = 0.02
MIN_SPEED = 0.005
CYCLE_PERIOD = 1.0


class _Origin:
    """Point3D-like, comme le champ `origin` de SparseVoxelGrid."""

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _GridMsg:
    """La géométrie d'un SparseVoxelGrid, pour ReachabilityModel.check_grid."""

    def __init__(self, size, resolution, origin):
        self.size_x, self.size_y, self.size_z = size
        self.resolution = resolution
        self.origin = origin


class Harness:
    """Évalue Q(dx, dy) pour des déplacements virtuels de la base."""

    def __init__(self, group='0', package=PACKAGE, precision=PRECISION, verbose=True):
        self.verbose = verbose
        self.model = ReachabilityModel(package, precision, fallback=True,
                                       log=print if verbose else (lambda _msg: None))
        self.device = self.model.device
        self.backend = self.model.backend

        with h5py.File(MAPS_H5, 'r') as f:
            grp = f['group'][group]
            attrs = dict(grp.attrs)
            # Candidat 4 = offset (0,0) : la scène non translatée.
            obst = np.array(grp['candidate']['4']['obstacle_map'])
            self.recorded_rm = np.array(grp['candidate']['4']['reachability_map'])
            self.recorded_mask = np.array(grp['candidate']['4']['mask'])
            self.recorded_q = (np.array(grp['q_scores'])
                               if 'q_scores' in grp else None)
        self.attrs = attrs

        self.res = float(attrs['voxel_size'])
        self.size = (int(attrs['voxel_grid_size_x']),
                     int(attrs['voxel_grid_size_y']),
                     int(attrs['voxel_grid_size_z']))
        self.origin = _Origin(float(attrs['origin_x']),
                              float(attrs['origin_y']),
                              float(attrs['origin_z']))
        self.vg_info = {
            'origin': self.origin, 'resolution': self.res,
            'size_x': self.size[0], 'size_y': self.size[1], 'size_z': self.size[2],
        }

        # La carte stockée est l'entrée déjà encodée de l'ancien chemin (+1 obstacle,
        # -1 libre) : on en reprend l'occupation, que le Package encode à sa façon.
        self.scene = torch.as_tensor(obst > 0, device=self.device).unsqueeze(0).unsqueeze(0)
        mismatch = self.model.check_grid(_GridMsg(self.size, self.res, self.origin))
        if mismatch:
            raise ValueError(f'{MAPS_H5}: {mismatch}')

        # Centre de la région = centroïde du masque enregistré (la position du
        # marqueur/goal de ce cycle), reconverti en mètres.
        idx = np.argwhere(self.recorded_mask)
        ci, cj, ck = idx.mean(axis=0)
        self.center = (self.origin.x + ci * self.res,
                       self.origin.y + cj * self.res,
                       self.origin.z + ck * self.res)

        self.max_batch = CANDIDATES
        if verbose:
            print(f'[harness] group={group} package={self.model.package.name} '
                  f'backend={self.backend} device={self.device}')
            print(f'[harness] res={self.res:.4f} m  origin=({self.origin.x:.2f},'
                  f'{self.origin.y:.2f},{self.origin.z:.2f})  size={self.size}')
            print(f'[harness] centre région = ({self.center[0]:.3f}, '
                  f'{self.center[1]:.3f}, {self.center[2]:.3f}) m')
            print(f'[harness] q_center enregistré = {attrs.get("q_center")}')

    # ── Géométrie ────────────────────────────────────────────────────────────

    def shifted_scene(self, dx, dy):
        """Carte d'obstacles vue depuis une base déplacée de (dx, dy).

        Reproduit ObstacleMapTransformer.transform : décalage entier, voxels
        entrants libres (voir §1.6 du rapport : c'était -1 dans l'entrée encodée).
        """
        di = -int(round(dx / self.res))
        dj = -int(round(dy / self.res))
        return translate_filled(self.scene, (0, 0, di, dj), fill_value=False), di, dj

    def entering_mask(self, di, dj):
        """Masque booléen des voxels créés par la translation (sans source)."""
        m = torch.ones_like(self.scene, dtype=torch.bool)
        return translate_filled(m, (0, 0, di, dj), fill_value=False).logical_not()

    def base_mask(self, centers=None, radii=None):
        """Masque union de sphères à la position non translatée."""
        centers = centers if centers is not None else [self.center]
        radii = radii if radii is not None else [WORKSPACE_RADIUS] * len(centers)
        return WorkspaceEvaluation(centers, radii, self.device).region_mask(**self.vg_info)

    def mask_for(self, base_mask, di, dj):
        """Masque de région du candidat (dx,dy), par décalage entier — comme
        GradientBasedController.compute_quality_scores."""
        return base_mask if (di == 0 and dj == 0) else base_mask.translated(di, dj)

    # ── Évaluation ───────────────────────────────────────────────────────────

    def q_at(self, offsets, base_mask=None, return_rm=False, chunk=None):
        """Q pour chaque offset (dx, dy), en mètres, par paquets de `chunk` Scenes.

        Args:
            offsets: liste de (dx, dy)
            base_mask: VoxelMask de la région non translatée (construit si None)
            return_rm: renvoie aussi les cartes de reachability (coûteux en RAM)
            chunk: Scenes par appel à predict (défaut: les 9 candidats du contrôleur ;
                le Package les passe une par une au réseau)

        Returns:
            np.ndarray des Q, et (si return_rm) la liste des cartes
        """
        base_mask = base_mask if base_mask is not None else self.base_mask()
        chunk = chunk or self.max_batch

        qs, rms = [], []
        for s in range(0, len(offsets), chunk):
            batch = offsets[s:s + chunk]
            tensors, shifts = [], []
            for dx, dy in batch:
                t, di, dj = self.shifted_scene(dx, dy)
                tensors.append(t)
                shifts.append((di, dj))
            maps = self.model.predict(torch.cat(tensors, dim=0)[:, 0])
            for n, (di, dj) in enumerate(shifts):
                mask = self.mask_for(base_mask, di, dj)
                qs.append(mask.compute_mean(maps[n]))
                if return_rm:
                    rms.append(maps[n].cpu().numpy())
            del maps
        return (np.array(qs), rms) if return_rm else np.array(qs)

    # ── Reproduction exacte du contrôleur ────────────────────────────────────

    @staticmethod
    def grid_offsets(delta=DELTA, n=GRID_SIZE):
        """Identique à GradientBasedController.grid_offsets()."""
        half = n // 2
        return [(((i % n) - half) * delta, (half - (i // n)) * delta)
                for i in range(n * n)]

    @staticmethod
    def fit_plane(scores, delta=DELTA, n=GRID_SIZE):
        """Gradient moindres carrés, formule exacte du code de production.

        Renvoie (gx, gy, c, residu) — le résidu n'est PAS calculé en production,
        il est ajouté ici pour la mesure.
        """
        offs = Harness.grid_offsets(delta, n)
        x = np.array([o[0] for o in offs])
        y = np.array([o[1] for o in offs])
        s = np.asarray(scores, dtype=float)
        gx = float((x * s).sum() / (x ** 2).sum())
        gy = float((y * s).sum() / (y ** 2).sum())
        c = float(s.mean())
        pred = c + gx * x + gy * y
        dof = max(1, len(s) - 3)
        return gx, gy, c, float(np.sqrt(((s - pred) ** 2).sum() / dof))

    @staticmethod
    def fit_quadratic(scores, delta=DELTA, n=GRID_SIZE):
        """Ajustement quadratique 6 coefficients sur le même voisinage.

        Renvoie (coeffs, residu, H) avec Q ≈ c + gx·x + gy·y + a·x² + b·y² + d·xy
        et H = [[2a, d], [d, 2b]].
        """
        offs = Harness.grid_offsets(delta, n)
        x = np.array([o[0] for o in offs])
        y = np.array([o[1] for o in offs])
        s = np.asarray(scores, dtype=float)
        A = np.column_stack([np.ones_like(x), x, y, x ** 2, y ** 2, x * y])
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        pred = A @ coef
        dof = max(1, len(s) - 6)
        resid = float(np.sqrt(((s - pred) ** 2).sum() / dof))
        H = np.array([[2 * coef[3], coef[5]], [coef[5], 2 * coef[4]]])
        return coef, resid, H

    @staticmethod
    def command(gx, gy, gain=CONTROL_GAIN, vmax=BASE_SPEED, vmin=MIN_SPEED):
        """Reproduit gradient_to_velocity + la zone morte de BaseCommander.

        Renvoie (vx, vy, regime) avec regime ∈ {deadband, proportional, saturated}.
        """
        vx, vy = gain * gx, gain * gy
        mag = float(np.hypot(vx, vy))
        regime = 'proportional'
        if mag > vmax:
            vx, vy = vx * vmax / mag, vy * vmax / mag
            regime = 'saturated'
        if float(np.hypot(vx, vy)) < vmin:
            return 0.0, 0.0, 'deadband'
        return vx, vy, regime
