#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright: Fiona Sunderland 2022, fes1g16@soton.ac.uk
Release 2022-10: Preliminary model featuring mean shape and four modes of varition.

The following files should be downloaded prior to running this script:
Size Normalised Mean Limb shape (Mean_Limb_Shape.stl)
Full Statistical shape model modes (Modes.npy)
Linear Regression Model (LR.pkl)

The script requires the module ampscan.
ampscan can be found at https://github.com/abel-research/ampscan
Installation details can be found at https://ampscan.readthedocs.io/en/latest/

This script allows for generation of synthetic residual limbs using the eigenvectors (modes) from the statistical shape model.
The eigenvalues (components) for each eigenvector controls the variation of each model's shape from the mean.

This script generates 100 possible values for each eigenvalue which are evenly spaced between the max and min 
eigenvalues for each mode found in the training data. It then perturbs the mean shape by this amount in each eigenvector (mode)

As such, in this preliminary model, 100^10 possible generations exist. 
However, at second release (2023-06) it is recommended that mode 6 and mode 7 should not be used together. 
Later modes may be biased by the small training sample so can be commented out but each can be introduced 
to the random generation model if desired by uncommenting any of lines 95-100. This will be resolved in a future release.

The current PCA model produces some mode combinations that result in limbs that are not anatomically possible
To combat this a skin-only PCA model has been produced to generate the initial random shape
Then, a linear regression model is used to predict full model mode scores from the skin-only model mode scores
The predicted eigenvalues and full model eigenvectors are then used to generate an anatomically feasible model

The random generated model is in a size-normalised state.
To create a 'real' size model a uniform scaling should be applied.
To do this line 103 can be uncommented and (scale factor) replaced with the scale factor desired
The scale factor represents the desired length of the intact (unamputated) tibia for the subject
Suggested value is 383 (average of the training data in mm), though this can also be randomised (line 102).
The training data estimated intact tibia length range was 342.8 - 439.8 mm.

Include:
- path to full mode eigenvectors (components) as described in line 79
- path to size normalised mean shape should be included in line 57
- path to linear regression pickled model should be included in line 74, and
- save path desired as described in line 128/131

"""

import numpy as np
import matplotlib.pyplot as plt
from ampscan import AmpObject
from ampscan.vis import vtkRenWin
import os
import random
import copy
import pickle
import argparse


def GenerateRandomLimb(file_path, n_samples, start, save_mesh):
    mean = AmpObject(
        '/opt/app/OpenLimbTT/version-2023-06/Mean_Limb_Shape.stl', unify=False)

    component = np.zeros([10, n_samples])

    # component[0] = random.choice(np.linspace(-14.80692194113721 , 25.4537448635005, 100))
    # component[1] = random.choice(np.linspace(-5.37869110537926 , 10.06971435469401, 100))
    # component[2] = random.choice(np.linspace(-3.996990835549319 , 4.12168595787859, 100))
    # component[3] = random.choice(np.linspace(-4.05537984190567 , 4.24308399616669, 100))
    # component[4] = random.choice(np.linspace(-2.403525754650053 , 4.73074159745632, 100))
    # component[5] = random.choice(np.linspace(-1.854646894835898 , 2.390247129446665, 100))
    # component[6] = random.choice(np.linspace(-2.13215613028021 , 2.19420856255569, 100))
    # component[7] = random.choice(np.linspace(-1.197319576810893 , 1.221847275562656, 100))
    # component[8] = random.choice(np.linspace(-0.798154129842514 , 0.802807003549, 100))
    # component[9] = random.choice(np.linspace(-0.7353096205329 , 0.95283973472981, 100))

    minima = np.array([
        -14.80692194113721,
        -5.37869110537926,
        -3.996990835549319,
        -4.05537984190567,
        -2.403525754650053,
        -1.854646894835898,
        -2.13215613028021,
        -1.197319576810893,
        -0.798154129842514,
        -0.7353096205329
    ])

    maxima = np.array([
        25.4537448635005,
        10.06971435469401,
        4.12168595787859,
        4.24308399616669,
        4.73074159745632,
        2.390247129446665,
        2.19420856255569,
        1.221847275562656,
        0.802807003549,
        0.95283973472981
    ])

    component = scale_range(np.random.rand(n_samples, 10), minima, maxima)

    pickled_model = pickle.load(
        open('/opt/app/OpenLimbTT/version-2023-06/LR.pkl', 'rb'))
    newcomponent = pickled_model.predict(component)


    X = np.load('/opt/app/OpenLimbTT/version-2023-06/Components.npy')

    verts = copy.deepcopy(mean.vert)[None] + np.sum((newcomponent[..., None]
                                                     * X[None]).reshape(n_samples, 10, mean.vert.shape[0], 3), axis=1)


    # scale factor =  random.choice(np.linspace(342.8, 439.8 , 100))
    # synthetic.vert = (synthetic.vert)* (scale factor)

    if not file_path.endswith("/"):
        file_path += "/"

    np.save(file_path + f"components_{start:05d}.npy", newcomponent)
    synthetic = copy.deepcopy(mean)
    # np.save("mean_vert.npy", mean.vert)
    # np.save("raw_components.npy", X)

    if save_mesh:
        for i, vert_pos in enumerate(verts):
            synthetic.vert = vert_pos
            synthetic.save(file_path + "limb_{:05d}.stl".format(i + start))


def scale_range(x, min, max):
    return x*(max - min) + min


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_limbs', type=int, default=1,
                        help='Number of limbs to generate')
    parser.add_argument("--path", type=str, default="./stls/",
                        help="Path to output folder")
    parser.add_argument("--start", type=int, default=0,
                        help="Number to start labelling from")
    parser.add_argument("--save_mesh", type=int, default=1, help="Generate .stl files if true")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    args = parser.parse_args()

    np.random.seed(args.seed)

    GenerateRandomLimb(args.path, args.num_limbs, args.start, args.save_mesh)
