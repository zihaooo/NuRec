# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nurec.models.gaussians.strategies.base import BaseGaussianStrategy
from nurec.models.gaussians.strategies.gsplat import GSplatStrategy
from nurec.models.gaussians.strategies.mcmc import MCMCStrategy


try:
    from nurec.models.gaussians.strategies.ecad import ECADStrategy
except ModuleNotFoundError as error:
    # Bazel runfiles built before ecad.py existed do not contain it; everything else still works without ECAD.
    if error.name != "nurec.models.gaussians.strategies.ecad":
        raise
    ECADStrategy = None  # type: ignore[assignment,misc]


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseGaussianStrategy",
    "ECADStrategy",
    "GSplatStrategy",
    "MCMCStrategy",
]
