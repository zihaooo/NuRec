# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Road floor for densification (mcity multiclip, 2026-11-12): no Gaussians under the road the rig drove on.

The road surface lies a fixed height under the rig origin (mcity 80 s, measured on a trained model: 0.81 m). A 2D grid
over the drive stores, for every cell within `corridor` of the drive line (horizontal distance to the nearest rig
pose), the road height under that nearest pose; a Gaussian whose centre lies more than `margin` below the road height
of its cell is "under the road". Cells farther from the drive have no road height (NaN), so nothing there is ever
flagged: away from the road the ground can really be lower (ditches, grass, slopes).

Works in the NuRec frame, which must keep the world's z axis up (world_to_nurec = translation and uniform scale).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def rig_origins_in_nurec(rig_trajectories) -> tuple[np.ndarray, float]:
    """All rig origins of a RigTrajectories in the NuRec frame [N, 3], and the NuRec units per world metre."""
    m = np.asarray(rig_trajectories.world_to_nurec.matrix, dtype=np.float64)
    rot, trans, w = m[:3, :3], m[:3, 3], float(m[3, 3])
    scale = float(np.linalg.norm(rot[:, 0])) / w
    up = rot @ np.array([0.0, 0.0, 1.0])
    if not np.allclose(up / np.linalg.norm(up), [0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("road_floor needs a NuRec frame whose z axis is the world's up axis")
    origins = np.concatenate(
        [t.T_rig_worlds[:, :3, 3].detach().cpu().double().numpy() for t in rig_trajectories.rig_trajectories]
    )
    return (origins @ rot.T + trans) / w, scale


class RoadFloorGrid:
    """Road height per grid cell near the drive (NaN elsewhere), queried by Gaussian centre."""

    def __init__(self, x0: float, y0: float, cell: float, zgrid: torch.Tensor) -> None:
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.zgrid = zgrid  # [H, W] float32, NaN outside the corridor

    @classmethod
    def from_rig_origins(
        cls, origins: np.ndarray, below: float, corridor: float, cell: float, chunk: int = 2000
    ) -> "RoadFloorGrid":
        """origins [N, 3] in the NuRec frame (z up); below, corridor, cell in NuRec units."""
        if len(origins) == 0:
            raise ValueError("road_floor: no rig poses")
        keep = [0]  # thin the poses to one per half cell of travel (stops and 100 Hz poses add nothing)
        last = origins[0, :2]
        for k in range(1, len(origins)):
            if np.hypot(*(origins[k, :2] - last)) >= 0.5 * cell:
                keep.append(k)
                last = origins[k, :2]
        o = origins[keep]
        x0, y0 = o[:, 0].min() - corridor - cell, o[:, 1].min() - corridor - cell
        w = int(math.ceil((o[:, 0].max() + corridor + cell - x0) / cell)) + 1
        h = int(math.ceil((o[:, 1].max() + corridor + cell - y0) / cell)) + 1
        r = int(math.ceil(corridor / cell)) + 1
        di, dj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="xy")
        di, dj = di.ravel(), dj.ravel()
        best_d = np.full(h * w, np.inf)
        best_z = np.full(h * w, np.nan)
        for s in range(0, len(o), chunk):
            p = o[s : s + chunk]
            ci = np.floor((p[:, 0] - x0) / cell).astype(np.int64)[:, None] + di[None]
            cj = np.floor((p[:, 1] - y0) / cell).astype(np.int64)[:, None] + dj[None]
            d = np.hypot(x0 + (ci + 0.5) * cell - p[:, :1], y0 + (cj + 0.5) * cell - p[:, 1:2])
            z = np.broadcast_to(p[:, 2:3] - below, d.shape)
            ok = (d <= corridor) & (ci >= 0) & (ci < w) & (cj >= 0) & (cj < h)
            flat, d, z = (cj * w + ci)[ok], d[ok], z[ok]
            order = np.lexsort((d, flat))  # per cell, nearest pose first
            flat, d, z = flat[order], d[order], z[order]
            first = np.ones(len(flat), dtype=bool)
            first[1:] = flat[1:] != flat[:-1]
            flat, d, z = flat[first], d[first], z[first]
            better = d < best_d[flat]
            best_d[flat[better]] = d[better]
            best_z[flat[better]] = z[better]
        return cls(x0, y0, cell, torch.from_numpy(best_z.reshape(h, w).astype(np.float32)))

    def to(self, device: torch.device | str) -> "RoadFloorGrid":
        return RoadFloorGrid(self.x0, self.y0, self.cell, self.zgrid.to(device))

    @property
    def n_cells(self) -> int:
        return int(torch.isfinite(self.zgrid).sum().item())

    @torch.no_grad()
    def road_z(self, positions: torch.Tensor) -> torch.Tensor:
        """Road height under each position [N] (NaN off the corridor or off the grid)."""
        h, w = self.zgrid.shape
        ix = torch.floor((positions[:, 0].float() - self.x0) / self.cell).long()
        iy = torch.floor((positions[:, 1].float() - self.y0) / self.cell).long()
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out = torch.full((positions.shape[0],), float("nan"), device=positions.device)
        out[inside] = self.zgrid[iy[inside], ix[inside]]
        return out

    @torch.no_grad()
    def under(self, positions: torch.Tensor, margin: float) -> torch.Tensor:
        """Boolean [N]: the position lies more than `margin` below the road of its cell (False off the corridor)."""
        z = self.road_z(positions)
        return torch.isfinite(z) & (positions[:, 2].float() < z - margin)


class RoadHeightField:
    """Whole-scene companion of RoadFloorGrid (2026-11-13): on a coarse grid reaching `extent` beyond the drive's
    bounding box, every cell holds the horizontal distance to the nearest rig pose and the road height under it;
    positions beyond the grid take the nearest edge cell (they are far from the drive either way)."""

    def __init__(self, x0: float, y0: float, cell: float, road: torch.Tensor, dist: torch.Tensor) -> None:
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.road, self.dist = road, dist  # [H, W] float32 each

    @classmethod
    def from_rig_origins(
        cls, origins: np.ndarray, below: float, extent: float, cell: float, chunk: int = 8192
    ) -> "RoadHeightField":
        keep = [0]  # poses one cell of travel apart
        last = origins[0, :2]
        for k in range(1, len(origins)):
            if np.hypot(*(origins[k, :2] - last)) >= cell:
                keep.append(k)
                last = origins[k, :2]
        o = origins[keep]
        x0, y0 = o[:, 0].min() - extent, o[:, 1].min() - extent
        w = int(math.ceil((o[:, 0].max() + extent - x0) / cell)) + 1
        h = int(math.ceil((o[:, 1].max() + extent - y0) / cell)) + 1
        cx = (x0 + (np.arange(w) + 0.5) * cell)[None, :].repeat(h, 0).ravel()
        cy = (y0 + (np.arange(h) + 0.5) * cell)[:, None].repeat(w, 1).ravel()
        dist = np.empty(h * w)
        road = np.empty(h * w)
        for s in range(0, h * w, chunk):
            dd = np.hypot(cx[s : s + chunk, None] - o[None, :, 0], cy[s : s + chunk, None] - o[None, :, 1])
            j = dd.argmin(1)
            dist[s : s + chunk] = dd[np.arange(len(j)), j]
            road[s : s + chunk] = o[j, 2] - below
        f = lambda a: torch.from_numpy(a.reshape(h, w).astype(np.float32))  # noqa: E731
        return cls(x0, y0, cell, f(road), f(dist))

    def to(self, device: torch.device | str) -> "RoadHeightField":
        return RoadHeightField(self.x0, self.y0, self.cell, self.road.to(device), self.dist.to(device))

    @torch.no_grad()
    def query(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(road height under the nearest pose, horizontal distance to it) per position, each [N]."""
        h, w = self.road.shape
        ix = torch.floor((positions[:, 0].float() - self.x0) / self.cell).long().clamp(0, w - 1)
        iy = torch.floor((positions[:, 1].float() - self.y0) / self.cell).long().clamp(0, h - 1)
        return self.road[iy, ix], self.dist[iy, ix]


@torch.no_grad()
def out_of_bounds(
    positions: torch.Tensor,
    floor: RoadFloorGrid,
    margin: float,
    field: Optional[RoadHeightField] = None,
    near_margin: Optional[float] = None,
    near_distance: float = 0.0,
    far_margin: Optional[float] = None,
    ceiling: Optional[float] = None,
) -> dict[str, torch.Tensor]:
    """Boolean masks [N] of the rule parts: 'road' (corridor, as RoadFloorGrid.under), 'ground' (below the road by more
    than the near / far margin outside the corridor), 'ceiling' (higher than `ceiling` above the road anywhere), and
    'any'. All lengths in NuRec units."""
    out = {"road": floor.under(positions, margin)}
    z = positions[:, 2].float()
    ground = torch.zeros_like(out["road"])
    high = torch.zeros_like(out["road"])
    if field is not None and (near_margin is not None or far_margin is not None or ceiling is not None):
        road, dist = field.query(positions)
        off_corridor = ~torch.isfinite(floor.road_z(positions))
        inf = float("inf")
        band = torch.where(
            dist <= near_distance,
            torch.full_like(dist, near_margin if near_margin is not None else inf),
            torch.full_like(dist, far_margin if far_margin is not None else inf),
        )
        ground = off_corridor & (z < road - band)
        if ceiling is not None:
            high = z > road + ceiling
    out["ground"], out["ceiling"] = ground, high
    out["any"] = out["road"] | ground | high
    return out
