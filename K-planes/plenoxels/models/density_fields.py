"""
Density proposal field
"""
import itertools
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
import logging as log

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.models.kplane_field import normalize_aabb
from plenoxels.ops.interpolation import grid_sample_wrapper
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


class KPlaneDensityField(nn.Module):
    def __init__(self,
                 aabb,
                 resolution,
                 num_input_coords,
                 num_output_coords,
                 density_activation: Callable,
                 spatial_distortion: Optional[SpatialDistortion] = None,
                 linear_decoder: bool = True):
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.hexplane = num_input_coords == 4
        self.feature_dim = num_output_coords
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder
        activation = "ReLU"
        if self.linear_decoder:
            activation = "None"

        self.grids = self.init_grid_param(out_dim=num_output_coords, reso=resolution)

        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        log.info(f"Initialized KPlaneDensityField. hexplane={self.hexplane} - "
                 f"resolution={resolution}")
        log.info(f"KPlaneDensityField grids: \n{self.grids}")

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None and self.hexplane:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = self.interpolate_ms_features(
            pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None)
        density = self.density_activation(
            self.sigma_net(features).to(pts)
            #features.to(pts)
        ).view(n_rays, n_samples, 1)
        return density

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = {k: v for k, v in self.sigma_net.named_parameters(prefix="sigma_net")}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
    

    def interpolate_ms_features(self, pts: torch.Tensor,
                                ms_grids,
                                grid_dimensions: int,
                                concat_features,
                                num_levels: Optional[int],
                                ) -> torch.Tensor:
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]), grid_dimensions)
        )
        if num_levels is None:
            num_levels = len(ms_grids)
        multi_scale_interp = [] if concat_features else 0.
        grid: nn.ParameterList
        for scale_id, grid in enumerate(ms_grids[:num_levels]):
            interp_space = 1.
            for ci, coo_comb in enumerate(coo_combs):
                # interpolate in plane
                feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                interp_out_plane = (
                    grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                    .view(-1, feature_dim)
                )
                # compute product over planes
                interp_space = interp_space * interp_out_plane # maybe can modify to add or connect

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp

    def init_grid_param(self,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
        coo_combs = list(itertools.combinations(range(4), 2))
        grid_coefs = nn.ParameterList()
        for ci, coo_comb in enumerate(coo_combs):
            new_grid_coef = nn.Parameter(torch.empty(
                [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]] #  Dynamic models (in_dim == 4) will have 6 planes: # (y, x), (z, x), (t, x), (z, y), (t, y), (t, z) # static models (in_dim == 3) will only have the 1st, 2nd and 4th planes.
            ))
            if 3 in coo_comb:  # Initialize time planes to 1
                nn.init.ones_(new_grid_coef)
            else:
                nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)
        return grid_coefs
    
