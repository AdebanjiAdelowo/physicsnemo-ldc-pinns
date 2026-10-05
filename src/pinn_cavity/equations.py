# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The class NavierStokes is copied from examples/cfd/ldc_pinns/train.py of NVIDIA
# PhysicsNeMo, commit b45a5c810c741e6b41f8515be24c51121f8fc21f, without changes to
# its body. Modified by Adebanji Adelowo (2026): moved into a module of its own with
# the imports it needs; `viscosity` added.

from physicsnemo.sym.eq.pde import PDE
from sympy import Function, Number, Symbol


class NavierStokes(PDE):
    """Incompressible Navier-Stokes equations (steady, 2D).

    Simplified from the compressible form in physicsnemo-sym for the case
    where ``rho`` is constant and ``time=False``.

    Reference: https://turbmodels.larc.nasa.gov/implementrans.html
    """

    def __init__(self, nu=0.01, rho=1.0, dim=2, time=False):
        self.dim = dim
        x, y = Symbol("x"), Symbol("y")
        iv = {"x": x, "y": y}
        u = Function("u")(*iv.values())
        v = Function("v")(*iv.values())
        p = Function("p")(*iv.values())
        nu, rho = Number(nu), Number(rho)
        self.equations = {
            "continuity": u.diff(x) + v.diff(y),
            "momentum_x": (
                u * u.diff(x)
                + v * u.diff(y)
                + (1 / rho) * p.diff(x)
                - nu * u.diff(x, 2)
                - nu * u.diff(y, 2)
            ),
            "momentum_y": (
                u * v.diff(x)
                + v * v.diff(y)
                + (1 / rho) * p.diff(y)
                - nu * v.diff(x, 2)
                - nu * v.diff(y, 2)
            ),
        }


def viscosity(reynolds: float, lid_velocity: float, width: float) -> float:
    """Kinematic viscosity for ``Re = lid_velocity * width / nu``."""
    if reynolds <= 0:
        raise ValueError(f"the Reynolds number must be positive, got {reynolds}")
    return lid_velocity * width / reynolds
