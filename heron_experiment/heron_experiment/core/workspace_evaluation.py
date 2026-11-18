#!/usr/bin/env python3
"""Score a reachability map over a workspace region.

The region is a set of spheres: one centre scores a single goal (the ArTag, or
an MPC goal), several score the approach corridor the arm actually traverses
(goal + the tail of the MPC path). Every sphere weighs exactly 1/N, so Q is by
construction the average of the per-sphere means — see WeightedVoxelMask, which
replaced a boolean union whose per-sphere weight drifted with how much the
spheres overlapped.

This is the single "region -> Q" abstraction in the package. `gradient_controller`
builds one instance per region and reuses it across cycles for its mask cache;
`brain` and `workspace_reachability_node` call `compute_quality_score` directly.
"""

import torch

from .voxel_mask import WeightedVoxelMask, get_coord_grids


class WorkspaceEvaluation:
    """
    Evaluate reachability quality score within a spherical workspace region.

    The region is a set of spheres of `radius` around one or more centres, each
    weighing 1/N in the score. center_z is meaningful (unlike the previous
    cylindrical definition).

    All operations stay on GPU to avoid CPU-GPU transfers. The region mask is
    cached against the voxel grid geometry, so a static target rebuilds nothing
    between cycles.
    """

    def __init__(self, centers_xyz, radius=0.30, device='cuda'):
        """
        Args:
            centers_xyz: a single (x, y, z) tuple, or an iterable of them, in
                         meters — the sphere centres of the region
            radius: float sphere radius shared by every centre (default: 0.30m),
                or a sequence of floats matching centers_xyz (one radius per
                centre — e.g. a larger sphere for the goal than for the path tail)
            device: torch device ('cuda' or 'cpu')
        """
        self.device = device
        self._centers = self._normalize_centers(centers_xyz)
        self._radius = self._normalize_radius(radius, len(self._centers))

        self._cached_mask = None
        self._cache_key = None

    @staticmethod
    def _normalize_centers(centers_xyz):
        """Accept either one (x, y, z) or an iterable of them; return a list."""
        items = list(centers_xyz)
        if not items:
            raise ValueError('Workspace region needs at least one center')
        # A bare (x, y, z) has scalar elements; a region has tuple elements.
        if not hasattr(items[0], '__len__'):
            items = [items]
        return [tuple(float(v) for v in c) for c in items]

    @staticmethod
    def _normalize_radius(radius, n_centers):
        """Accept either one shared radius or a per-centre sequence; return a list."""
        if hasattr(radius, '__len__'):
            radii = [float(r) for r in radius]
            if len(radii) != n_centers:
                raise ValueError(
                    f'radius sequence length {len(radii)} != {n_centers} centers')
            return radii
        return [float(radius)] * n_centers

    @property
    def centers(self):
        """The region centres, as a list of (x, y, z) tuples."""
        return self._centers

    @property
    def radius(self):
        """Sphere radius in meters, one per centre (see _normalize_radius)."""
        return self._radius

    def region_mask(self, origin, resolution, size_x, size_y, size_z):
        """
        Equal-weight sphere mask for this region on the given voxel grid.

        Cached against the grid geometry: the same instance reused across cycles
        rebuilds the mask only when the grid itself changes.

        Args:
            origin, resolution, size_x, size_y, size_z: voxel grid parameters

        Returns:
            WeightedVoxelMask: the region mask
        """
        cache_key = (origin.x, origin.y, origin.z, resolution, size_x, size_y, size_z)
        if self._cached_mask is not None and self._cache_key == cache_key:
            return self._cached_mask

        mask = WeightedVoxelMask.equal_weight_union_of_spheres(
            self._centers, self._radius,
            origin, resolution, size_x, size_y, size_z,
            device=self.device
        )

        self._cached_mask = mask
        self._cache_key = cache_key
        return mask

    def compute_quality_score(self, reachability_map, origin, resolution, size_x, size_y, size_z):
        """
        Compute quality score Q: equal-weight mean reachability over the region
        spheres (see WeightedVoxelMask).

        Args:
            reachability_map: torch.Tensor on GPU, shape (size_x, size_y, size_z)
            origin, resolution, size_x, size_y, size_z: voxel grid parameters

        Returns:
            float: Mean reachability value inside the region
        """
        mask = self.region_mask(origin, resolution, size_x, size_y, size_z)
        return mask.compute_mean(reachability_map)

    @staticmethod
    def compute_quality_scores_batched(
            reachability_map, centers, origin, resolution,
            size_x, size_y, size_z, radius, device='cuda'):
        """
        Compute mean reachability inside a sphere of `radius` around each center.

        Distinct from the combined region above: this returns *one score per
        centre*, used to profile a whole MPC path pose by pose. The voxel
        coordinate grids are shared (see voxel_mask.get_coord_grids) instead of
        rebuilt per centre.

        Stays a plain boolean mean per sphere — it divides by the voxels present
        in the grid, where the combined region divides by the unclipped sphere
        count. The two therefore differ for a sphere the grid edge cuts through.

        Args:
            reachability_map: torch.Tensor on GPU, shape (size_x, size_y, size_z)
            centers: iterable of (x, y, z) tuples in meters
            origin, resolution, size_x, size_y, size_z: voxel grid parameters
            radius: float, sphere radius in meters
            device: torch device ('cuda' or 'cpu')

        Returns:
            list[float]: mean reachability for each center, 0.0 for an empty sphere
        """
        i_grid, j_grid, k_grid = get_coord_grids(size_x, size_y, size_z, device)
        radius_voxels_sq = (radius / resolution) ** 2

        scores = []
        for cx, cy, cz in centers:
            center_i = (cx - origin.x) / resolution
            center_j = (cy - origin.y) / resolution
            center_k = (cz - origin.z) / resolution

            dist_sq = ((i_grid - center_i) ** 2 + (j_grid - center_j) ** 2
                       + (k_grid - center_k) ** 2)
            mask = dist_sq <= radius_voxels_sq

            count = torch.sum(mask)
            if count.item() > 0:
                score = (torch.sum(reachability_map[mask]) / count).item()
            else:
                score = 0.0
            scores.append(score)

        return scores
