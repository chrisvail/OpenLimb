import torch

from ..geometry.measure import Measure
from ..geometry.sampler import OpenLimbSampler


class GaussNewton:
    def __init__(self, measure: Measure, sampler: OpenLimbSampler, n_components: int, component_weight=1E-2, eps=1E-4):
        self.measure = measure
        self.sampler = sampler
        self.n_components = n_components
        self.component_weight = component_weight
        self.eps = eps

    def __call__(self, measurements, iterations=20, initialisation="mean"):
        if initialisation == "mean":
            initialisation = torch.zeros(
                (len(measurements), self.n_components), dtype=torch.float64, device=measurements.device)
        elif initialisation == "random":
            initialisation = self.sampler.normalise_components(
                self.sampler.sample(len(measurements)))
        else:
            raise ValueError(
                f"Invalid initialisation (mean | random): {initialisation}.")

        return torch.stack(
            [self._process_one(init, m.double(), iterations)
             for m, init in zip(measurements, initialisation, strict=True)],
            dim=0
        )

    @torch.no_grad()
    def _process_one(self, components, measurements, iterations):
        pred_measurements = self.measure(
            self.sampler.to_mesh(components, True)[0], True, True).squeeze()
        error = (measurements - pred_measurements).pow(2).sum() \
            + self.component_weight * components.pow(2).sum()
        lambda_ = 1

        for _ in range(iterations):
            jac = self._jacobian(components)

            new_error = torch.inf
            failed_step = False
            residual = (pred_measurements - measurements)
            while new_error > error:
                new_components = components - torch.linalg.solve(
                    jac.T@jac + (lambda_ + self.component_weight) *
                    torch.eye(self.n_components, dtype=components.dtype,
                              device=components.device),
                    jac.T @ residual + self.component_weight *
                    components
                )

                new_pred_measurements = self.measure(
                    self.sampler.to_mesh(new_components, True)[0], True, True).squeeze()

                new_error = (measurements - new_pred_measurements).pow(2).sum() \
                    + self.component_weight * new_components.pow(2).sum()

                if new_error > error:
                    lambda_ *= 2

                    if lambda_ > 1E8:
                        failed_step = True
                        break
                else:
                    lambda_ /= 8

            if failed_step:
                break

            components = new_components
            error = new_error
            pred_measurements = new_pred_measurements

        return components

    def _jacobian(self, components):
        variation = self.eps * torch.concat(
            (torch.eye(self.n_components, dtype=components.dtype, device=components.device),
             -torch.eye(self.n_components, dtype=components.dtype, device=components.device)))

        components_to_measure = components + variation
        verts = self.sampler.to_mesh(components_to_measure, True)[0]
        measurements = self.measure(verts, True, True)

        return ((measurements[:self.n_components] -
                measurements[self.n_components:]) / (2*self.eps)).T
