"""Gradient-free fitting of the 11 OpenLimbTT numbers to a set of seven measurements.

* ga_notebook        - the GA exactly as configured in genetic_alg.ipynb (what GA_results*.csv contain)
* lm_notebook_fixed  - the notebook's Levenberg-Marquardt `step`, with its faults fixed
* ga_improved        - same library (pygad), same objective, same budget; different search space and operators
* gn_polish          - projected Gauss-Newton inside the plausible box (finite-difference Jacobian)

Everything measures limbs with FastMeasure, which returns the same numbers as SSM_Driver.measure.
"""
import time

import numpy as np
import pygad
import torch

from .fast_measure import BoxMap, FastMeasure
from .geometry import load_transforms

N_GENES = 11


class Problem:
    """Measurement function in the two coordinate systems used below, with an evaluation counter."""

    def __init__(self, fm: FastMeasure = None):
        self.fm = fm or FastMeasure()
        self.box = BoxMap()
        comp_tf, _ = load_transforms(True)
        self.mean, self.std = comp_tf[0], comp_tf[1]
        self.n_evals = 0

    def measure(self, raw):
        self.n_evals += len(raw)
        return self.fm(raw)

    def from_z(self, x):          # z-scored components (the notebook's genes) -> raw
        return torch.as_tensor(np.asarray(x)).double().reshape(-1, N_GENES) * self.std + self.mean

    def from_u(self, x):          # box coordinates in [-1, 1] -> raw
        return self.box.to_raw(torch.as_tensor(np.asarray(x)).reshape(-1, N_GENES))

    def fitness(self, raw, target):
        """pygad maximises, so this is minus SSM_Driver.MeasurementLoss (mean squared *relative* error)."""
        m = self.measure(raw)
        return -((m - target) ** 2 / (target ** 2 + 1e-12)).mean(1)


# ----------------------------------------------------------------------------------- the notebook, as is
def ga_notebook(prob: Problem, target, seed):
    """genetic_alg.ipynb cell 4, unchanged except that a generation is measured in one batched call."""
    ga = pygad.GA(
        num_generations=50, num_parents_mating=4, sol_per_pop=32, num_genes=N_GENES,
        fitness_func=lambda ga, sols, idx: prob.fitness(prob.from_z(sols), target).tolist(), fitness_batch_size=32,
        gene_space={"low": -3, "high": 3}, parent_selection_type="sss", keep_parents=2, mutation_percent_genes=20,
        crossover_type="single_point", mutation_type="random", random_seed=seed, suppress_warnings=True)
    ga.run()
    best, _, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
    return np.asarray(best, dtype=np.float64)       # z-scored components


def lm_notebook_fixed(prob: Problem, target, x0, steps=10, lambda_=1e-3, delta=1e-6):
    """The notebook's `step`, repaired. Its three faults (so the recorded results contain no refinement at all):

    1. `func` was `dataset.get_measures`, which expects raw components, but `x` holds z-scored ones;
    2. the finite-difference Jacobian was reshaped (batch, measures, components) instead of
       (batch, components, measures) transposed, and had its sign flipped;
    3. the refined limb was printed but `best_solution` (unrefined) was what got written to the CSV.
    """
    func = lambda x: prob.measure(prob.from_z(x))
    x = torch.as_tensor(np.asarray(x0)).double().reshape(1, N_GENES)
    eye = torch.eye(N_GENES, dtype=torch.float64) * delta
    try:
        for _ in range(steps):
            J = ((func(x + eye) - func(x - eye)) / (2 * delta)).T[None]          # (1, measures, components)
            residuals = (target - func(x)).reshape(1, -1, 1)
            JTJ = J.transpose(1, 2) @ J
            damping = torch.diag_embed(torch.diagonal(JTJ, dim1=1, dim2=2))
            JTr = J.transpose(1, 2) @ residuals
            x_new = x + torch.linalg.solve(JTJ + lambda_ * damping, JTr).squeeze(-1)
            while (residuals ** 2).sum() < ((target - func(x_new)) ** 2).sum():
                lambda_ *= 10
                x_new = x + torch.linalg.solve(JTJ + lambda_ * damping, JTr).squeeze(-1)
                if lambda_ > 1e10:
                    raise ValueError()
            x, lambda_ = x_new, lambda_ / 10
    except ValueError:
        pass
    return x.squeeze(0).numpy()


# ----------------------------------------------------------------------------------- improved
def ga_improved(prob: Problem, target, seed, pop=32, gens=50, parents=8, elite=2, sigma0=0.3, sigma1=0.03,
                p_mut=0.3, crossover="blend_gene", selection="sss"):
    """Same objective and the same 1600-measurement budget as the notebook. What changed:

    * genes are the generator's box coordinates u in [-1, 1]^11, so every candidate is a plausible limb
      (the notebook searched z-scored components in [-3, 3]^11, most of which the generator can never produce);
    * mutation is a Gaussian step on each gene with probability p_mut, whose size shrinks geometrically from
      sigma0 to sigma1 (in box half-widths) over the run. The notebook's replaces 2 of 11 genes by a fresh
      uniform draw from the whole range in every child, at every generation, so it can never settle;
    * per-gene blend crossover (each child gene a random mix of the two parents' genes), 2 elites.
    Defaults chosen on 48 development limbs (not the test limbs): see claude/report/WRITEUP.md section 8.3.
    """
    rng = np.random.default_rng(seed)

    def mutate(offspring, ga):
        sigma = sigma0 * (sigma1 / sigma0) ** (ga.generations_completed / max(gens - 1, 1))
        step = rng.normal(0, sigma, offspring.shape) * (rng.random(offspring.shape) < p_mut)
        return np.clip(offspring + step, -1, 1)

    def blend(par, size, ga):
        i, j = rng.integers(0, len(par), size[0]), rng.integers(0, len(par), size[0])
        w = rng.random((size[0], 1)) if crossover == "blend" else rng.random(size)
        return w * par[i] + (1 - w) * par[j]

    ga = pygad.GA(
        num_generations=gens, num_parents_mating=parents, sol_per_pop=pop, num_genes=N_GENES,
        fitness_func=lambda ga, sols, idx: prob.fitness(prob.from_u(sols), target).tolist(), fitness_batch_size=pop,
        init_range_low=-1, init_range_high=1, parent_selection_type=selection, keep_elitism=elite,
        crossover_type=blend if crossover in ("blend", "blend_gene") else crossover, mutation_type=mutate,
        random_seed=seed, suppress_warnings=True)
    ga.run()
    best, _, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
    return np.asarray(best, dtype=np.float64)       # box coordinates


def gn_polish(prob: Problem, target, u0, iters=30, h=1e-4, tol_mm=1e-6):
    """Projected Gauss-Newton inside the plausible box. 7 equations, 11 unknowns, so each step is the
    minimum-norm step that zeroes the linearised residual (pinv), using only the genes not pinned at a bound;
    it is clipped to the box and halved until the error drops. Jacobian = 22 extra measurements, one batched call."""
    tgt = target.numpy()
    res = lambda u: (prob.measure(prob.from_u(u)).numpy() - tgt) / tgt       # relative error, (k, 7)
    lim = 1 - 1e-12
    u = np.clip(np.asarray(u0, dtype=np.float64), -1, 1)
    r = res(u)[0]
    for _ in range(iters):
        if np.abs(r * tgt).max() < tol_mm:
            break
        m = res(u[None] + h * np.concatenate([np.eye(N_GENES), -np.eye(N_GENES)]))
        J = ((m[:N_GENES] - m[N_GENES:]) / (2 * h)).T                           # (7, 11)
        free = np.ones(N_GENES, bool)
        for _ in range(N_GENES):
            step = np.zeros(N_GENES)
            step[free] = -np.linalg.pinv(J[:, free]) @ r
            pinned = free & (((u >= lim) & (step > 0)) | ((u <= -lim) & (step < 0)))
            if not pinned.any():
                break
            free &= ~pinned
        alpha, improved = 1.0, False
        while alpha > 1e-4:
            un = np.clip(u + alpha * step, -1, 1)
            rn = res(un)[0]
            if (rn ** 2).sum() < (r ** 2).sum():
                improved = True
                break
            alpha /= 2
        if not improved:
            break
        u, r = un, rn
    return u


# ----------------------------------------------------------------------------------- variants
def solve(prob: Problem, variant, target, seed, start=None, **kw):
    """One request -> {stage: (raw components (11,), measurements used so far, seconds so far)}.
    Stages after the first continue from the previous one, so each run reports "GA" and "GA + refinement"."""
    t0, out = time.perf_counter(), {}
    note = lambda name, raw: out.__setitem__(name, (raw, prob.n_evals, time.perf_counter() - t0))
    if variant == "notebook":
        x = ga_notebook(prob, target, seed)
        note("GA (notebook)", prob.from_z(x)[0])
        note("GA (notebook) + LM (fixed)", prob.from_z(lm_notebook_fixed(prob, target, x))[0])
    elif variant == "improved":
        u = ga_improved(prob, target, seed, **kw)
        note("GA (improved)", prob.from_u(u)[0])
        note("GA (improved) + GN", prob.from_u(gn_polish(prob, target, u))[0])
    elif variant in ("gn", "gn_random"):              # no GA: Gauss-Newton from the box centre / a random plausible limb
        u0 = np.zeros(N_GENES) if start is None else start
        note("GN (box centre)" if variant == "gn" else "GN (random start)", prob.from_u(gn_polish(prob, target, u0))[0])
    else:
        raise ValueError(variant)
    return out


_PROB = None


def _init_worker():
    global _PROB
    torch.set_num_threads(1)
    _PROB = Problem()


def _work(job):
    variant, i, target, seed, start, kw = job
    _PROB.n_evals = 0
    res = solve(_PROB, variant, torch.as_tensor(target).double(), seed, start=start, **dict(kw))
    return i, {st: (raw.numpy(), ev, sec) for st, (raw, ev, sec) in res.items()}


def run(pool, variant, targets, seeds, starts=None, **kw):
    """All requests of one variant through a multiprocessing pool (one request per task, one thread per worker).
    -> {stage: dict(raw (N, 11), evals (N,), secs (N,))}; secs are single-thread with other workers running."""
    jobs = [(variant, i, np.asarray(t), int(seeds[i]), None if starts is None else starts[i], tuple(kw.items()))
            for i, t in enumerate(targets)]
    out = {}
    for i, res in pool.imap_unordered(_work, jobs, chunksize=1):
        for st, (raw, ev, sec) in res.items():
            d = out.setdefault(st, dict(raw=np.zeros((len(jobs), N_GENES)), evals=np.zeros(len(jobs), int),
                                        secs=np.zeros(len(jobs))))
            d["raw"][i], d["evals"][i], d["secs"][i] = raw, ev, sec
    return out
