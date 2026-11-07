from pathlib import Path

import torch


class OpenLimbSampler:
    def __init__(self, batch_size, seed, data_dir: Path, scale: bool = True):
        self.scale = scale
        self.batch_size = batch_size

        # Random prime = 3581
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed*3581)

        # Linear Regression Map
        lr = torch.load(data_dir / "lr_map.pt")
        self.coeffs = lr["coef"].double()
        self.intercepts = lr["intercept"].double()

        self.n_modes = self.intercepts.shape[0]

        self._setup_component_stats(data_dir)
        self._setup_mesh_components(data_dir)

    def _setup_component_stats(self, data_dir):
        component_ranges = torch.load(data_dir / "component_range.pt")
        self.mins = torch.concat(
            (component_ranges["skin_min"], component_ranges["scale_min"][None])) if self.scale else component_ranges["skin_min"]
        self.maxes = torch.concat(
            (component_ranges["skin_max"], component_ranges["scale_max"][None])) if self.scale else component_ranges["skin_max"]

        self.mean = self.coeffs @ (component_ranges["skin_max"] +
                                   component_ranges["skin_min"]) / 2 + self.intercepts
        self.std = torch.sqrt(torch.diag(self.coeffs @ torch.diag(
            (component_ranges["skin_max"] - component_ranges["skin_min"])**2 / 12) @ self.coeffs.T))

        if self.scale:
            self.mean = torch.concat(
                (self.mean, ((self.maxes[-1] + self.mins[-1]) / 2)[None]))
            self.std = torch.concat(
                (self.std, ((self.maxes[-1] - self.mins[-1]) / (12**0.5))[None]))

    def _setup_mesh_components(self, data_dir):
        mesh = torch.load(data_dir / "mesh.pt")
        self.mean_verts = mesh["mean_verts"].double()
        self.mode_basis = mesh["vert_components"].double()
        self.faces = mesh["faces"]
        self.selected_verts = mesh["selected_verts"]

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        components = self._rescale_components(torch.rand(
            (batch_size, self.n_modes + int(self.scale)), generator=self.generator, dtype=torch.float64))

        components[:, :self.n_modes] = components[:, :self.n_modes] @ self.coeffs.T + \
            self.intercepts

        return components

    def _rescale_components(self, components):
        return self.mins + components * (self.maxes - self.mins)

    def normalise_components(self, components):
        return (components - self.mean) / self.std

    def denormalise_components(self, components):
        return (components * self.std) + self.mean

    def get_plausibility_transform(self, normalised=False):
        center = (self.mins + self.maxes) / 2
        half_range = (self.maxes - self.mins) / 2

        mean = self.mean if normalised else torch.zeros_like(self.mean)
        std = self.std if normalised else torch.ones_like(self.std)

        mat = torch.diag(
            std[:self.n_modes]) @ torch.inverse(self.coeffs.T) @ torch.diag(1/half_range[:self.n_modes])

        vec = ((mean[:self.n_modes] -
               self.intercepts)@torch.inverse(self.coeffs.T) - center[:self.n_modes]) / half_range[:self.n_modes]

        if self.scale:
            scaled_mat = torch.eye(self.n_modes + 1, dtype=self.std.dtype) * \
                std[-1] / half_range[-1]
            scaled_vec = (mean - center) / half_range

            scaled_mat[:self.n_modes, :self.n_modes] = mat
            scaled_vec[:self.n_modes] = vec

            return scaled_mat, scaled_vec
        return mat, vec

    def to_mesh(self, components, normalised=False):
        no_batch = components.dim() == 1
        components = components.double().reshape(-1, self.n_modes + int(self.scale))
        if normalised:
            components = self.denormalise_components(components)

        modes = components[:, :self.n_modes]
        verts = self.mean_verts + \
            (modes @ self.mode_basis).reshape(-1, *self.mean_verts.shape)
        if self.scale:
            verts = verts * components[:, self.n_modes:, None]

        # Drop batch dim if given no batch
        return (verts[0] if no_batch else verts), self.faces
