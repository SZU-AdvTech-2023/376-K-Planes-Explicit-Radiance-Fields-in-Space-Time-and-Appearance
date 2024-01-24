import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.ops.interpolation import grid_sample_wrapper
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


def positional_encoding(positions, freqs):
    freq_bands = 2**torch.arange(freqs,
                                 dtype=torch.float16,
                                 device=positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def min_max_quantize(inputs, bits=8):
    if bits == 32:
        return inputs

    scale = torch.amax(torch.abs(inputs)).clamp(min=1e-6)
    n = float(2**(bits - 1) - 1)
    out = torch.round(torch.abs(inputs / scale) * n) / n * scale
    rounded = out * torch.sign(inputs)
    # detach这一步就是STE (Straight Through Estimator)
    return (rounded - inputs).detach() + inputs


class KPlaneField(nn.Module):

    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        use_appearance_embedding: bool,
        appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        num_images: Optional[int],
        use_mask=False,
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        self.use_mask = use_mask

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [r * res for r in config["resolution"][:3]
                                    ] + config["resolution"][3:]
            gp = self.init_grid_param(
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
        log.info(f"Initialized model grids: {self.grids}")

        # 1. Init masks
        # if self.use_mask:
        #     self.masks = nn.ModuleList()
        #     for idx, res in enumerate(self.multiscale_res_multipliers):
        #         gp = self.init_mask_param(idx)
        #         self.masks.append(gp)
        #     log.info(f"Initialized model masks: {self.masks}")

        # 1. only space mask
        if self.use_mask:
            self.masks = nn.ModuleList()
            for idx, res in enumerate(self.multiscale_res_multipliers):
                gp = self.init_mask_param(idx)
                self.masks.append(gp)
            log.info(f"Initialized model masks: {self.masks}")

        # 2. Init appearance code-related parameters
        self.use_average_appearance_embedding = True  # for test-time
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images
        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = nn.Embedding(
                self.num_images, self.appearance_embedding_dim)
        else:
            self.appearance_embedding_dim = 0

        # 3. Init decoder params
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        # 3. Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.
                appearance_embedding_dim,  #self.direction_encoder.n_output_dims,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",  #original
                    # "activation": "Tanh",
                    "output_activation": "None",
                    "n_neurons": 64,  # original
                    "n_hidden_layers": 1,  # original
                },
            )
            # self.in_dim_color = (self.direction_encoder.n_output_dims +
            #                      self.geo_feat_dim + 48 +
            #                      self.appearance_embedding_dim)
            self.in_dim_color = (self.direction_encoder.n_output_dims +
                                 self.geo_feat_dim +
                                 self.appearance_embedding_dim)  # original
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",  # original
                    # "activation": "Tanh",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,  # original
                    # "n_hidden_layers": 2,
                    "n_hidden_layers": 8,
                },
            )

    def get_density(self,
                    pts: torch.Tensor,
                    timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(
                -1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps),
                            dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = self.interpolate_ms_features(
            pts,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            num_levels=None)
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(
                features, [self.geo_feat_dim, 1], dim=-1)

        density = self.density_activation(
            density_before_activation.to(pts)).view(n_rays, n_samples, 1)
        return density, features

    def forward(self,
                pts: torch.Tensor,
                directions: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        camera_indices = None
        if self.use_appearance_embedding:
            if timestamps is None:
                raise AttributeError(
                    "timestamps (appearance-ids) are not provided.")
            camera_indices = timestamps
            timestamps = None
        density, features = self.get_density(pts, timestamps)
        n_rays, n_samples = pts.shape[:2]

        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)
        if not self.linear_decoder:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)

        if self.linear_decoder:
            color_features = [features]
        else:
            if timestamps is not None:
                timestamps = timestamps[:, None].expand(
                    -1, n_samples)[..., None]  # [n_rays, n_samples, 1]
                pts = torch.cat((pts, timestamps),
                                dim=-1)  # [n_rays, n_samples, 4]
            # PE_feature = positional_encoding(pts, freqs=6)
            # color_features = [
            #     encoded_directions,
            #     features.view(-1, self.geo_feat_dim),
            #     torch.squeeze(PE_feature, dim=1)
            # ]
            color_features = [
                encoded_directions,
                features.view(-1, self.geo_feat_dim),
            ]

        if self.use_appearance_embedding:
            if camera_indices.dtype == torch.float32:
                # Interpolate between two embeddings. Currently they are hardcoded below.
                #emb1_idx, emb2_idx = 100, 121  # trevi
                emb1_idx, emb2_idx = 11, 142  # sacre
                emb_fn = self.appearance_embedding
                emb1 = emb_fn(
                    torch.full_like(camera_indices, emb1_idx,
                                    dtype=torch.long))
                emb1 = emb1.view(emb1.shape[0], emb1.shape[2])
                emb2 = emb_fn(
                    torch.full_like(camera_indices, emb2_idx,
                                    dtype=torch.long))
                emb2 = emb2.view(emb2.shape[0], emb2.shape[2])
                embedded_appearance = torch.lerp(emb1, emb2, camera_indices)
            elif self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                if hasattr(self, "test_appearance_embedding"):
                    embedded_appearance = self.test_appearance_embedding(
                        camera_indices)
                elif self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1],
                         self.appearance_embedding_dim),
                        device=directions.device
                    ) * self.appearance_embedding.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1],
                         self.appearance_embedding_dim),
                        device=directions.device)

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = embedded_appearance.view(
                -1, 1, ea_dim).expand(n_rays, n_samples,
                                      -1).reshape(-1, ea_dim)
            if not self.linear_decoder:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)

        if self.linear_decoder:
            if self.use_appearance_embedding:
                basis_values = self.color_basis(
                    torch.cat([directions, embedded_appearance], dim=-1))
            else:
                basis_values = self.color_basis(
                    directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(
                color_features.shape[0], 3,
                -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values,
                            dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(
                n_rays, n_samples, 3)

        return {"rgb": rgb, "density": density}

    def init_grid_param(self,
                        out_dim: int,
                        reso: Sequence[int],
                        a: float = 0.1,
                        b: float = 0.5):
        coo_combs = list(itertools.combinations(range(4), 2))
        grid_coefs = nn.ParameterList()
        for ci, coo_comb in enumerate(coo_combs):
            new_grid_coef = nn.Parameter(
                torch.empty([1, out_dim] + [
                    reso[cc] for cc in coo_comb[::-1]
                ]  #  Dynamic models (in_dim == 4) will have 6 planes: # (y, x), (z, x), (t, x), (z, y), (t, y), (t, z) # static models (in_dim == 3) will only have the 1st, 2nd and 4th planes.
                            ))
            if 3 in coo_comb:  # Initialize time planes to 1
                nn.init.ones_(new_grid_coef)
            else:
                nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)
        return grid_coefs

    # def init_mask_param(self, res_index):
    #     mask_coefs = nn.ParameterList()
    #     for i in range(6):
    #         new_mask_coef = nn.Parameter(
    #             torch.ones_like(self.grids[res_index][i]))
    #         mask_coefs.append(new_mask_coef)
    #     return mask_coefs

    # only mask space
    def init_mask_param(self, res_index):
        mask_coefs = nn.ParameterList()
        for i in range(6):
            if i != 2 and i != 4 and i != 5:
                new_mask_coef = nn.Parameter(
                    torch.ones_like(self.grids[res_index][i]))
                mask_coefs.append(new_mask_coef)
        return mask_coefs

    def interpolate_ms_features(
        self,
        pts: torch.Tensor,
        grid_dimensions: int,
        num_levels: Optional[int],
    ) -> torch.Tensor:
        coo_combs = list(
            itertools.combinations(range(pts.shape[-1]), grid_dimensions))
        if num_levels is None:
            num_levels = len(self.grids)
        multi_scale_interp = [] if self.concat_features else 0.
        grid: nn.ParameterList

        for scale_id, grid in enumerate(self.grids[:num_levels]):
            t = 0
            interp_space = 1.
            # interp_space_cat = []
            for ci, coo_comb in enumerate(coo_combs):
                # plane = min_max_quantize(grid[ci], 8)
                plane = grid[ci]
                if self.use_mask:
                    if ci!=2 and ci!=4 and ci!=5:
                        mask = torch.sigmoid(self.masks[scale_id][t])
                        t+=1
                        plane = (plane * (mask >= 0.5) - plane * mask).detach() + plane * mask

                # interpolate in plane
                feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                # interp_out_plane = (
                #     grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                #     .view(-1, feature_dim)
                # )
                interp_out_plane = (grid_sample_wrapper(
                    plane, pts[..., coo_comb]).view(-1, feature_dim)
                                    )  # (N_samples, feature_dim)
                # compute product over planes
                interp_space = interp_space * interp_out_plane  # maybe can modify to concatenate

            # combine over scales
            if self.concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if self.concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp

    def get_params(self):
        field_params = {
            k: v
            for k, v in self.grids.named_parameters(prefix="grids")
        }
        if self.use_mask:
            field_masks_params = {
                k: v
                for k, v in self.masks.named_parameters(prefix="masks")
            }
        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(
                prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(
                self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(
                self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {
            k: v
            for k, v in self.named_parameters()
            if (k not in nn_params.keys() and k not in field_params.keys() and
                (self.use_mask and k not in field_masks_params.keys()))
        }

        if self.use_mask:
            return {
                "nn": list(nn_params.values()),
                "field": list(field_params.values()),
                "field_masks": list(field_masks_params.values()),
                "other": list(other_params.values()),
            }
        else:
            return {
                "nn": list(nn_params.values()),
                "field": list(field_params.values()),
                "other": list(other_params.values()),
            }

    def compact_save(self):
        """Save main field as a compressed hashmap
        """

        fp = './fields/'
        data = {}
        for i in range(6):
            if self.is_static:
                if i in [0, 1, 3]:
                    data[f'{i}'] = self.kplanes[i].compact_save()
            else:
                data[f'{i}'] = self.kplanes[i].compact_save()

        # Compress
        import pickle
        if self.compression_type == 'pickle':
            with open(f'{fp}pickle_field.pickle', 'wb') as f:
                pickle.dump(data, f)

        elif self.compression_type == 'GZIP':
            import gzip
            with gzip.open(f"{fp}gzip_field.gz", "wb") as f:
                pickle.dump(data, f)

        elif self.compression_type == 'BZ2':
            import bz2
            with bz2.BZ2File(f'{fp}bz2_field.pbz2', 'wb') as f:
                pickle.dump(data, f)

        elif self.compression_type == 'LZMA':
            import lzma
            with lzma.open(f"{fp}lzma_field.xz", "wb") as f:
                pickle.dump(data, f)

    def compact_load(self):
        """Load compressed model
        """
        fp = './fields/'
        import pickle
        if self.compression_type == 'pickle':
            with open(f'{fp}pickle_field.pickle', 'rb') as handle:
                dictionary = pickle.load(handle)
        elif self.compression_type == 'GZIP':
            import gzip
            with gzip.open(f"{fp}gzip_field.gz", "rb") as f:
                dictionary = pickle.load(f)

        elif self.compression_type == 'BZ2':
            import bz2
            with bz2.BZ2File(f'{fp}bz2_field.pbz2', 'rb') as f:
                dictionary = pickle.load(f)

        elif self.compression_type == 'LZMA':
            import lzma
            with lzma.open(f"{fp}lzma_field.xz", "rb") as f:
                dictionary = pickle.load(f)

        print(f'Loading Grids ...')
        from tqdm import tqdm
        for i in tqdm(range(6)):
            if self.is_static:
                if i in [0, 1, 3]:
                    self.kplanes[i].compact_load(dictionary[f'{i}'])
            else:
                self.kplanes[i].compact_load(dictionary[f'{i}'])
