# -*- coding: future_fstrings -*-
#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

import numpy as np
import os
from .acados_model import AcadosModel
from .utils import get_acados_path, J_to_idx, J_to_idx_slack, get_lib_ext

class AcadosOcpDims:
    """
    Class containing the dimensions of the optimal control problem.
    """
    def __init__(self):
        self.__nx      = None
        self.__nu      = None
        self.__nz      = 0
        self.__np      = 0
        self.__ny      = 0
        self.__ny_e    = 0
        self.__ny_0    = 0
        self.__nr      = 0
        self.__nr_e    = 0
        self.__nh      = 0
        self.__nh_e    = 0
        self.__nphi    = 0
        self.__nphi_e  = 0
        self.__nbx     = 0
        self.__nbx_0   = 0
        self.__nbx_e   = 0
        self.__nbu     = 0
        self.__nsbx    = 0
        self.__nsbx_e  = 0
        self.__nsbu    = 0
        self.__nsh     = 0
        self.__nsh_e   = 0
        self.__nsphi   = 0
        self.__nsphi_e = 0
        self.__ns      = 0
        self.__ns_e    = 0
        self.__ng      = 0
        self.__ng_e    = 0
        self.__nsg     = 0
        self.__nsg_e   = 0
        self.__nbxe_0  = None
        self.__N       = None


    @property
    def nx(self):
        """:math:`n_x` - number of states.
        Type: int; default: None"""
        return self.__nx

    @property
    def nz(self):
        """:math:`n_z` - number of algebraic variables.
        Type: int; default: 0"""
        return self.__nz

    @property
    def nu(self):
        """:math:`n_u` - number of inputs.
        Type: int; default: None"""
        return self.__nu

    @property
    def np(self):
        """:math:`n_p` - number of parameters.
        Type: int; default: 0"""
        return self.__np

    @property
    def ny(self):
        """:math:`n_y` - number of residuals in Lagrange term.
        Type: int; default: 0"""
        return self.__ny

    @property
    def ny_0(self):
        """:math:`n_{y}^0` - number of residuals in Mayer term.
        Type: int; default: 0"""
        return self.__ny_0

    @property
    def ny_e(self):
        """:math:`n_{y}^e` - number of residuals in Mayer term.
        Type: int; default: 0"""
        return self.__ny_e

    @property
    def nr(self):
        """:math:`n_{\pi}` - dimension of the image of the inner nonlinear function in positive definite constraints.
        Type: int; default: 0"""
        return self.__nr

    @property
    def nr_e(self):
        """:math:`n_{\pi}^e` - dimension of the image of the inner nonlinear function in positive definite constraints.
        Type: int; default: 0"""
        return self.__nr_e

    @property
    def nh(self):
        """:math:`n_h` - number of nonlinear constraints.
        Type: int; default: 0"""
        return self.__nh

    @property
    def nh_e(self):
        """:math:`n_{h}^e` - number of nonlinear constraints at terminal shooting node N.
        Type: int; default: 0"""
        return self.__nh_e

    @property
    def nphi(self):
        """:math:`n_{\phi}` - number of convex-over-nonlinear constraints.
        Type: int; default: 0"""
        return self.__nphi

    @property
    def nphi_e(self):
        """:math:`n_{\phi}^e` - number of convex-over-nonlinear constraints at terminal shooting node N.
        Type: int; default: 0"""
        return self.__nphi_e

    @property
    def nbx(self):
        """:math:`n_{b_x}` - number of state bounds.
        Type: int; default: 0"""
        return self.__nbx

    @property
    def nbxe_0(self):
        """:math:`n_{be_{x0}}` - number of state bounds at initial shooting node that are equalities.
        Type: int; default: None"""
        return self.__nbxe_0

    @property
    def nbx_0(self):
        """:math:`n_{b_{x0}}` - number of state bounds for initial state.
        Type: int; default: 0"""
        return self.__nbx_0

    @property
    def nbx_e(self):
        """:math:`n_{b_x}` - number of state bounds at terminal shooting node N.
        Type: int; default: 0"""
        return self.__nbx_e

    @property
    def nbu(self):
        """:math:`n_{b_u}` - number of input bounds.
        Type: int; default: 0"""
        return self.__nbu

    @property
    def nsbx(self):
        """:math:`n_{{sb}_x}` - number of soft state bounds.
        Type: int; default: 0"""
        return self.__nsbx

    @property
    def nsbx_e(self):
        """:math:`n_{{sb}^e_{x}}` - number of soft state bounds at terminal shooting node N.
        Type: int; default: 0"""
        return self.__nsbx_e

    @property
    def nsbu(self):
        """:math:`n_{{sb}_u}` - number of soft input bounds.
        Type: int; default: 0"""
        return self.__nsbu

    @property
    def nsg(self):
        """:math:`n_{{sg}}` - number of soft general linear constraints.
        Type: int; default: 0"""
        return self.__nsg

    @property
    def nsg_e(self):
        """:math:`n_{{sg}^e}` - number of soft general linear constraints at terminal shooting node N.
        Type: int; default: 0"""
        return self.__nsg_e

    @property
    def nsh(self):
        """:math:`n_{{sh}}` - number of soft nonlinear constraints.
        Type: int; default: 0"""
        return self.__nsh

    @property
    def nsh_e(self):
        """:math:`n_{{sh}}^e` - number of soft nonlinear constraints at terminal shooting node N.
        Type: int; default: 0"""
        return self.__nsh_e

    @property
    def nsphi(self):
        """:math:`n_{{s\phi}}` - number of soft convex-over-nonlinear constraints.
        Type: int; default: 0"""
        return self.__nsphi

    @property
    def nsphi_e(self):
        """:math:`n_{{s\phi}^e}` - number of soft convex-over-nonlinear constraints at terminal shooting node N.
        Type: int; default: 0"""
        return self.__nsphi_e

    @property
    def ns(self):
        """:math:`n_{s}` - total number of slacks.
        Type: int; default: 0"""
        return self.__ns

    @property
    def ns_e(self):
        """:math:`n_{s}^e` - total number of slacks at terminal shooting node N.
        Type: int; default: 0"""
        return self.__ns_e

    @property
    def ng(self):
        """:math:`n_{g}` - number of general polytopic constraints.
        Type: int; default: 0"""
        return self.__ng

    @property
    def ng_e(self):
        """:math:`n_{g}^e` - number of general polytopic constraints at terminal shooting node N.
        Type: int; default: 0"""
        return self.__ng_e

    @property
    def N(self):
        """:math:`N` - prediction horizon.
        Type: int; default: None"""
        return self.__N

    @nx.setter
    def nx(self, nx):
        if isinstance(nx, int) and nx > 0:
            self.__nx = nx
        else:
            raise Exception('Invalid nx value, expected positive integer.')

    @nz.setter
    def nz(self, nz):
        if isinstance(nz, int) and nz > -1:
            self.__nz = nz
        else:
            raise Exception('Invalid nz value, expected nonnegative integer.')

    @nu.setter
    def nu(self, nu):
        if isinstance(nu, int) and nu > -1:
            self.__nu = nu
        else:
            raise Exception('Invalid nu value, expected nonnegative integer.')

    @np.setter
    def np(self, np):
        if isinstance(np, int) and np > -1:
            self.__np = np
        else:
            raise Exception('Invalid np value, expected nonnegative integer.')

    @ny_0.setter
    def ny_0(self, ny_0):
        if isinstance(ny_0, int) and ny_0 > -1:
            self.__ny_0 = ny_0
        else:
            raise Exception('Invalid ny_0 value, expected nonnegative integer.')

    @ny.setter
    def ny(self, ny):
        if isinstance(ny, int) and ny > -1:
            self.__ny = ny
        else:
            raise Exception('Invalid ny value, expected nonnegative integer.')

    @ny_e.setter
    def ny_e(self, ny_e):
        if isinstance(ny_e, int) and ny_e > -1:
            self.__ny_e = ny_e
        else:
            raise Exception('Invalid ny_e value, expected nonnegative integer.')

    @nr.setter
    def nr(self, nr):
        if isinstance(nr, int) and nr > -1:
            self.__nr = nr
        else:
            raise Exception('Invalid nr value, expected nonnegative integer.')

    @nr_e.setter
    def nr_e(self, nr_e):
        if isinstance(nr_e, int) and nr_e > -1:
            self.__nr_e = nr_e
        else:
            raise Exception('Invalid nr_e value, expected nonnegative integer.')

    @nh.setter
    def nh(self, nh):
        if isinstance(nh, int) and nh > -1:
            self.__nh = nh
        else:
            raise Exception('Invalid nh value, expected nonnegative integer.')

    @nh_e.setter
    def nh_e(self, nh_e):
        if isinstance(nh_e, int) and nh_e > -1:
            self.__nh_e = nh_e
        else:
            raise Exception('Invalid nh_e value, expected nonnegative integer.')

    @nphi.setter
    def nphi(self, nphi):
        if isinstance(nphi, int) and nphi > -1:
            self.__nphi = nphi
        else:
            raise Exception('Invalid nphi value, expected nonnegative integer.')

    @nphi_e.setter
    def nphi_e(self, nphi_e):
        if isinstance(nphi_e, int) and nphi_e > -1:
            self.__nphi_e = nphi_e
        else:
            raise Exception('Invalid nphi_e value, expected nonnegative integer.')

    @nbx.setter
    def nbx(self, nbx):
        if isinstance(nbx, int) and nbx > -1:
            self.__nbx = nbx
        else:
            raise Exception('Invalid nbx value, expected nonnegative integer.')

    @nbxe_0.setter
    def nbxe_0(self, nbxe_0):
        if isinstance(nbxe_0, int) and nbxe_0 > -1:
            self.__nbxe_0 = nbxe_0
        else:
            raise Exception('Invalid nbxe_0 value, expected nonnegative integer.')

    @nbx_0.setter
    def nbx_0(self, nbx_0):
        if isinstance(nbx_0, int) and nbx_0 > -1:
            self.__nbx_0 = nbx_0
        else:
            raise Exception('Invalid nbx_0 value, expected nonnegative integer.')

    @nbx_e.setter
    def nbx_e(self, nbx_e):
        if isinstance(nbx_e, int) and nbx_e > -1:
            self.__nbx_e = nbx_e
        else:
            raise Exception('Invalid nbx_e value, expected nonnegative integer.')

    @nbu.setter
    def nbu(self, nbu):
        if isinstance(nbu, int) and nbu > -1:
            self.__nbu = nbu
        else:
            raise Exception('Invalid nbu value, expected nonnegative integer.')

    @nsbx.setter
    def nsbx(self, nsbx):
        if isinstance(nsbx, int) and nsbx > -1:
            self.__nsbx = nsbx
        else:
            raise Exception('Invalid nsbx value, expected nonnegative integer.')

    @nsbx_e.setter
    def nsbx_e(self, nsbx_e):
        if isinstance(nsbx_e, int) and nsbx_e > -1:
            self.__nsbx_e = nsbx_e
        else:
            raise Exception('Invalid nsbx_e value, expected nonnegative integer.')

    @nsbu.setter
    def nsbu(self, nsbu):
        if isinstance(nsbu, int) and nsbu > -1:
            self.__nsbu = nsbu
        else:
            raise Exception('Invalid nsbu value, expected nonnegative integer.')

    @nsg.setter
    def nsg(self, nsg):
        if isinstance(nsg, int) and nsg > -1:
            self.__nsg = nsg
        else:
            raise Exception('Invalid nsg value, expected nonnegative integer.')

    @nsg_e.setter
    def nsg_e(self, nsg_e):
        if isinstance(nsg_e, int) and nsg_e > -1:
            self.__nsg_e = nsg_e
        else:
            raise Exception('Invalid nsg_e value, expected nonnegative integer.')

    @nsh.setter
    def nsh(self, nsh):
        if isinstance(nsh, int) and nsh > -1:
            self.__nsh = nsh
        else:
            raise Exception('Invalid nsh value, expected nonnegative integer.')

    @nsh_e.setter
    def nsh_e(self, nsh_e):
        if isinstance(nsh_e, int) and nsh_e > -1:
            self.__nsh_e = nsh_e
        else:
            raise Exception('Invalid nsh_e value, expected nonnegative integer.')

    @nsphi.setter
    def nsphi(self, nsphi):
        if isinstance(nsphi, int) and nsphi > -1:
            self.__nsphi = nsphi
        else:
            raise Exception('Invalid nsphi value, expected nonnegative integer.')

    @nsphi_e.setter
    def nsphi_e(self, nsphi_e):
        if isinstance(nsphi_e, int) and nsphi_e > -1:
            self.__nsphi_e = nsphi_e
        else:
            raise Exception('Invalid nsphi_e value, expected nonnegative integer.')

    @ns.setter
    def ns(self, ns):
        if isinstance(ns, int) and ns > -1:
            self.__ns = ns
        else:
            raise Exception('Invalid ns value, expected nonnegative integer.')

    @ns_e.setter
    def ns_e(self, ns_e):
        if isinstance(ns_e, int) and ns_e > -1:
            self.__ns_e = ns_e
        else:
            raise Exception('Invalid ns_e value, expected nonnegative integer.')

    @ng.setter
    def ng(self, ng):
        if isinstance(ng, int) and ng > -1:
            self.__ng = ng
        else:
            raise Exception('Invalid ng value, expected nonnegative integer.')

    @ng_e.setter
    def ng_e(self, ng_e):
        if isinstance(ng_e, int) and ng_e > -1:
            self.__ng_e = ng_e
        else:
            raise Exception('Invalid ng_e value, expected nonnegative integer.')

    @N.setter
    def N(self, N):
        if isinstance(N, int) and N > 0:
            self.__N = N
        else:
            raise Exception('Invalid N value, expected positive integer.')

    def set(self, attr, value):
        setattr(self, attr, value)


class AcadosOcpCost:
    """
    Class containing the numerical data of the cost:

    NOTE: all cost terms, except for the terminal one are weighted with the corresponding time step.
    This means given the time steps are :math:`\Delta t_0,..., \Delta t_N`, the total cost is given by:
    :math:`c_\\text{total} = \Delta t_0 \cdot c_0(x_0, u_0, p_0, z_0) + ... + \Delta t_{N-1} \cdot c_{N-1}(x_0, u_0, p_0, z_0) + c_N(x_N, p_N)`.

    This means the Lagrange cost term is given in continuous time, this makes up for a seeminglessly OCP discretization with a nonuniform time grid.

    In case of LINEAR_LS:
    stage cost is
    :math:`l(x,u,z) = || V_x \, x + V_u \, u + V_z \, z - y_\\text{ref}||^2_W`,
    terminal cost is
    :math:`m(x) = || V^e_x \, x - y_\\text{ref}^e||^2_{W^e}`

    In case of NONLINEAR_LS:
    stage cost is
    :math:`l(x,u,z,p) = || y(x,u,z,p) - y_\\text{ref}||^2_W`,
    terminal cost is
    :math:`m(x,p) = || y^e(x,p) - y_\\text{ref}^e||^2_{W^e}`

    In case of CONVEX_OVER_NONLINEAR:
    stage cost is
    :math:`l(x,u,p) = \psi(y(x,u,p) - y_\\text{ref}, p)`,
    terminal cost is
    :math:`m(x, p) = \psi^e (y^e(x,p) - y_\\text{ref}^e, p)`
    """
    def __init__(self):
        # initial stage
        self.__cost_type_0 = None
        self.__W_0 = None
        self.__Vx_0 = None
        self.__Vu_0 = None
        self.__Vz_0 = None
        self.__yref_0 = None
        self.__cost_ext_fun_type_0 = 'casadi'
        # Lagrange term
        self.__cost_type   = 'LINEAR_LS'  # cost type
        self.__W           = np.zeros((0,0))
        self.__Vx          = np.zeros((0,0))
        self.__Vu          = np.zeros((0,0))
        self.__Vz          = np.zeros((0,0))
        self.__yref        = np.array([])
        self.__Zl          = np.array([])
        self.__Zu          = np.array([])
        self.__zl          = np.array([])
        self.__zu          = np.array([])
        self.__cost_ext_fun_type = 'casadi'
        # Mayer term
        self.__cost_type_e = 'LINEAR_LS'
        self.__W_e         = np.zeros((0,0))
        self.__Vx_e        = np.zeros((0,0))
        self.__yref_e      = np.array([])
        self.__Zl_e        = np.array([])
        self.__Zu_e        = np.array([])
        self.__zl_e        = np.array([])
        self.__zu_e        = np.array([])
        self.__cost_ext_fun_type_e = 'casadi'

    # initial stage
    @property
    def cost_type_0(self):
        """Cost type at initial shooting node (0)
        -- string in {EXTERNAL, LINEAR_LS, NONLINEAR_LS, CONVEX_OVER_NONLINEAR} or :code:`None`.
        Default: :code:`None`.

            .. note:: Cost at initial stage is the same as for intermediate shooting nodes if not set differently explicitly.

            .. note:: If :py:attr:`cost_type_0` is set to :code:`None` values in :py:attr:`W_0`, :py:attr:`Vx_0`, :py:attr:`Vu_0`, :py:attr:`Vz_0` and :py:attr:`yref_0` are ignored (set to :code:`None`).
        """
        return self.__cost_type_0

    @property
    def W_0(self):
        """:math:`W_0` - weight matrix at initial shooting node (0).
        Default: :code:`None`.
        """
        return self.__W_0

    @property
    def Vx_0(self):
        """:math:`V_x^0` - x matrix coefficient at initial shooting node (0).
        Default: :code:`None`.
        """
        return self.__Vx_0

    @property
    def Vu_0(self):
        """:math:`V_u^0` - u matrix coefficient at initial shooting node (0).
        Default: :code:`None`.
        """
        return self.__Vu_0

    @property
    def Vz_0(self):
        """:math:`V_z^0` - z matrix coefficient at initial shooting node (0).
        Default: :code:`None`.
        """
        return self.__Vz_0

    @property
    def yref_0(self):
        """:math:`y_\\text{ref}^0` - reference at initial shooting node (0).
        Default: :code:`None`.
        """
        return self.__yref_0

    @property
    def cost_ext_fun_type_0(self):
        """Type of external function for cost at initial shooting node (0)
        -- string in {casadi, generic} or :code:`None`
        Default: :code:'casadi'.

            .. note:: Cost at initial stage is the same as for intermediate shooting nodes if not set differently explicitly.
        """
        return self.__cost_ext_fun_type_0

    @yref_0.setter
    def yref_0(self, yref_0):
        if isinstance(yref_0, np.ndarray) and len(yref_0.shape) == 1:
            self.__yref_0 = yref_0
        else:
            raise Exception('Invalid yref_0 value, expected 1-dimensional numpy array.')

    @W_0.setter
    def W_0(self, W_0):
        if isinstance(W_0, np.ndarray) and len(W_0.shape) == 2:
            self.__W_0 = W_0
        else:
            raise Exception('Invalid cost W_0 value. ' \
                + 'Should be 2 dimensional numpy array.')

    @Vx_0.setter
    def Vx_0(self, Vx_0):
        if isinstance(Vx_0, np.ndarray) and len(Vx_0.shape) == 2:
            self.__Vx_0 = Vx_0
        else:
            raise Exception('Invalid cost Vx_0 value. ' \
                + 'Should be 2 dimensional numpy array.')

    @Vu_0.setter
    def Vu_0(self, Vu_0):
        if isinstance(Vu_0, np.ndarray) and len(Vu_0.shape) == 2:
            self.__Vu_0 = Vu_0
        else:
            raise Exception('Invalid cost Vu_0 value. ' \
                + 'Should be 2 dimensional numpy array.')

    @Vz_0.setter
    def Vz_0(self, Vz_0):
        if isinstance(Vz_0, np.ndarray) and len(Vz_0.shape) == 2:
            self.__Vz_0 = Vz_0
        else:
            raise Exception('Invalid cost Vz_0 value. ' \
                + 'Should be 2 dimensional numpy array.')

    @cost_ext_fun_type_0.setter
    def cost_ext_fun_type_0(self, cost_ext_fun_type_0):
        if cost_ext_fun_type_0 in ['casadi', 'generic']:
            self.__cost_ext_fun_type_0 = cost_ext_fun_type_0
        else:
            raise Exception('Invalid cost_ext_fun_type_0 value, expected numpy array.')

    # Lagrange term
    @property
    def cost_type(self):
        """
        Cost type at intermediate shooting nodes (1 to N-1)
        -- string in {EXTERNAL, LINEAR_LS, NONLINEAR_LS, CONVEX_OVER_NONLINEAR}.
        Default: 'LINEAR_LS'.
        """
        return self.__cost_type

    @property
    def W(self):
        """:math:`W` - weight matrix at intermediate shooting nodes (1 to N-1).
        Default: :code:`np.zeros((0,0))`.
        """
        return self.__W

    @property
    def Vx(self):
        """:math:`V_x` - x matrix coefficient at intermediate shooting nodes (1 to N-1).
        Default: :code:`np.zeros((0,0))`.
        """
        return self.__Vx

    @property
    def Vu(self):
        """:math:`V_u` - u matrix coefficient at intermediate shooting nodes (1 to N-1).
        Default: :code:`np.zeros((0,0))`.
        """
        return self.__Vu

    @property
    def Vz(self):
        """:math:`V_z` - z matrix coefficient at intermediate shooting nodes (1 to N-1).
        Default: :code:`np.zeros((0,0))`.
        """
        return self.__Vz

    @property
    def yref(self):
        """:math:`y_\\text{ref}` - reference at intermediate shooting nodes (1 to N-1).
        Default: :code:`np.array([])`.
        """
        return self.__yref

    @property
    def Zl(self):
        """:math:`Z_l` - diagonal of Hessian wrt lower slack at intermediate shooting nodes (0 to N-1).
        Default: :code:`np.array([])`.
        """
        return self.__Zl

    @property
    def Zu(self):
        """:math:`Z_u` - diagonal of Hessian wrt upper slack at intermediate shooting nodes (0 to N-1).
        Default: :code:`np.array([])`.
        """
        return self.__Zu

    @property
    def zl(self):
        """:math:`z_l` - gradient wrt lower slack at intermediate shooting nodes (0 to N-1).
        Default: :code:`np.array([])`.
        """
        return self.__zl

    @property
    def zu(self):
        """:math:`z_u` - gradient wrt upper slack at intermediate shooting nodes (0 to N-1).
        Default: :code:`np.array([])`.
        """
        return self.__zu

    @property
    def cost_ext_fun_type(self):
        """Type of external function for cost at intermediate shooting nodes (1 to N-1).
        -- string in {casadi, generic}
        Default: :code:'casadi'.
        """
        return self.__cost_ext_fun_type

    @cost_type.setter
    def cost_type(self, cost_type):
        cost_types = ('LINEAR_LS', 'NONLINEAR_LS', 'EXTERNAL', 'CONVEX_OVER_NONLINEAR')
        if cost_type in cost_types:
            self.__cost_type = cost_type
        else:
            raise Exception('Invalid cost_type value.')

    @cost_type_0.setter
    def cost_type_0(self, cost_type_0):
        cost_types = ('LINEAR_LS', 'NONLINEAR_LS', 'EXTERNAL', 'CONVEX_OVER_NONLINEAR')
        if cost_type_0 in cost_types:
            self.__cost_type_0 = cost_type_0
        else:
            raise Exception('Invalid cost_type_0 value.')

    @W.setter
    def W(self, W):
        if isinstance(W, np.ndarray) and len(W.shape) == 2:
            self.__W = W
        else:
            raise Exception('Invalid cost W value. ' \
                + 'Should be 2 dimensional numpy array.')


    @Vx.setter
    def Vx(self, Vx):
        if isinstance(Vx, np.ndarray) and len(Vx.shape) == 2:
            self.__Vx = Vx
        else:
            raise Exception('Invalid cost Vx value. ' \
                + 'Should be 2 dimensional numpy array.')

    @Vu.setter
    def Vu(self, Vu):
        if isinstance(Vu, np.ndarray) and len(Vu.shape) == 2:
            self.__Vu = Vu
        else:
            raise Exception('Invalid cost Vu value. ' \
                + 'Should be 2 dimensional numpy array.')

    @Vz.setter
    def Vz(self, Vz):
        if isinstance(Vz, np.ndarray) and len(Vz.shape) == 2:
            self.__Vz = Vz
        else:
            raise Exception('Invalid cost Vz value. ' \
                + 'Should be 2 dimensional numpy array.')

    @yref.setter
    def yref(self, yref):
        if isinstance(yref, np.ndarray) and len(yref.shape) == 1:
            self.__yref = yref
        else:
            raise Exception('Invalid yref value, expected 1-dimensional numpy array.')

    @Zl.setter
    def Zl(self, Zl):
        if isinstance(Zl, np.ndarray):
            self.__Zl = Zl
        else:
            raise Exception('Invalid Zl value, expected numpy array.')

    @Zu.setter
    def Zu(self, Zu):
        if isinstance(Zu, np.ndarray):
            self.__Zu = Zu
        else:
            raise Exception('Invalid Zu value, expected numpy array.')

    @zl.setter
    def zl(self, zl):
        if isinstance(zl, np.ndarray):
            self.__zl = zl
        else:
            raise Exception('Invalid zl value, expected numpy array.')

    @zu.setter
    def zu(self, zu):
        if isinstance(zu, np.ndarray):
            self.__zu = zu
        else:
            raise Exception('Invalid zu value, expected numpy array.')

    @cost_ext_fun_type.setter
    def cost_ext_fun_type(self, cost_ext_fun_type):
        if cost_ext_fun_type in ['casadi', 'generic']:
            self.__cost_ext_fun_type = cost_ext_fun_type
        else:
            raise Exception("Invalid cost_ext_fun_type value, expected one in ['casadi', 'generic'].")

    # Mayer term
    @property
    def cost_type_e(self):
        """
        Cost type at terminal shooting node (N)
        -- string in {EXTERNAL, LINEAR_LS, NONLINEAR_LS, CONVEX_OVER_NONLINEAR}.
        Default: 'LINEAR_LS'.
        """
        return self.__cost_type_e

    @property
    def W_e(self):
        """:math:`W_e` - weight matrix at terminal shooting node (N).
        Default: :code:`np.zeros((0,0))`.
        """
        return self.__W_e

    @property
    def Vx_e(self):
        """:math:`V_x^e` - x matrix coefficient for cost at terminal shooting node (N).
        Default: :code:`np.zeros((0,0))`.
        """
        return self.__Vx_e

    @property
    def yref_e(self):
        """:math:`y_\\text{ref}^e` - cost reference at terminal shooting node (N).
        Default: :code:`np.array([])`.
        """
        return self.__yref_e

    @property
    def Zl_e(self):
        """:math:`Z_l^e` - diagonal of Hessian wrt lower slack at terminal shooting node (N).
        Default: :code:`np.array([])`.
        """
        return self.__Zl_e

    @property
    def Zu_e(self):
        """:math:`Z_u^e` - diagonal of Hessian wrt upper slack at terminal shooting node (N).
        Default: :code:`np.array([])`.
        """
        return self.__Zu_e

    @property
    def zl_e(self):
        """:math:`z_l^e` - gradient wrt lower slack at terminal shooting node (N).
        Default: :code:`np.array([])`.
        """
        return self.__zl_e

    @property
    def zu_e(self):
        """:math:`z_u^e` - gradient wrt upper slack at terminal shooting node (N).
        Default: :code:`np.array([])`.
        """
        return self.__zu_e

    @property
    def cost_ext_fun_type_e(self):
        """Type of external function for cost at terminal shooting node (N).
        -- string in {casadi, generic}
        Default: :code:'casadi'.
        """
        return self.__cost_ext_fun_type_e

    @cost_type_e.setter
    def cost_type_e(self, cost_type_e):
        cost_types = ('LINEAR_LS', 'NONLINEAR_LS', 'EXTERNAL', 'CONVEX_OVER_NONLINEAR')

        if cost_type_e in cost_types:
            self.__cost_type_e = cost_type_e
        else:
            raise Exception('Invalid cost_type_e value.')

    @W_e.setter
    def W_e(self, W_e):
        if isinstance(W_e, np.ndarray) and len(W_e.shape) == 2:
            self.__W_e = W_e
        else:
            raise Exception('Invalid cost W_e value. ' \
                + 'Should be 2 dimensional numpy array.')

    @Vx_e.setter
    def Vx_e(self, Vx_e):
        if isinstance(Vx_e, np.ndarray) and len(Vx_e.shape) == 2:
            self.__Vx_e = Vx_e
        else:
            raise Exception('Invalid cost Vx_e value. ' \
                + 'Should be 2 dimensional numpy array.')

    @yref_e.setter
    def yref_e(self, yref_e):
        if isinstance(yref_e, np.ndarray) and len(yref_e.shape) == 1:
            self.__yref_e = yref_e
        else:
            raise Exception('Invalid yref_e value, expected 1-dimensional numpy array.')

    @Zl_e.setter
    def Zl_e(self, Zl_e):
        if isinstance(Zl_e, np.ndarray):
            self.__Zl_e = Zl_e
        else:
            raise Exception('Invalid Zl_e value, expected numpy array.')

    @Zu_e.setter
    def Zu_e(self, Zu_e):
        if isinstance(Zu_e, np.ndarray):
            self.__Zu_e = Zu_e
        else:
            raise Exception('Invalid Zu_e value, expected numpy array.')

    @zl_e.setter
    def zl_e(self, zl_e):
        if isinstance(zl_e, np.ndarray):
            self.__zl_e = zl_e
        else:
            raise Exception('Invalid zl_e value, expected numpy array.')

    @zu_e.setter
    def zu_e(self, zu_e):
        if isinstance(zu_e, np.ndarray):
            self.__zu_e = zu_e
        else:
            raise Exception('Invalid zu_e value, expected numpy array.')

    @cost_ext_fun_type_e.setter
    def cost_ext_fun_type_e(self, cost_ext_fun_type_e):
        if cost_ext_fun_type_e in ['casadi', 'generic']:
            self.__cost_ext_fun_type_e = cost_ext_fun_type_e
        else:
            raise Exception("Invalid cost_ext_fun_type_e value, expected one in ['casadi', 'generic'].")

    def set(self, attr, value):
        setattr(self, attr, value)


def print_J_to_idx_note():
    print("NOTE: J* matrix is converted to zero based vector idx* vector, which is returned here.")


class AcadosOcpConstraints:
    """
    class containing the description of the constraints
    """
    def __init__(self):
        self.__constr_type   = 'BGH'
        self.__constr_type_e = 'BGH'
        # initial x
        self.__lbx_0   = np.array([])
        self.__ubx_0   = np.array([])
        self.__idxbx_0 = np.array([])
        self.__idxbxe_0 = np.array([])
        # state bounds
        self.__lbx     = np.array([])
        self.__ubx     = np.array([])
        self.__idxbx   = np.array([])
        # bounds on x at shooting node N
        self.__lbx_e   = np.array([])
        self.__ubx_e   = np.array([])
        self.__idxbx_e = np.array([])
        # bounds on u
        self.__lbu     = np.array([])
        self.__ubu     = np.array([])
        self.__idxbu   = np.array([])
        # polytopic constraints
        self.__lg      = np.array([])
        self.__ug      = np.array([])
        self.__D       = np.zeros((0,0))
        self.__C       = np.zeros((0,0))
        # polytopic constraints at shooting node N
        self.__C_e     = np.zeros((0,0))
        self.__lg_e    = np.array([])
        self.__ug_e    = np.array([])
        # nonlinear constraints
        self.__lh      = np.array([])
        self.__uh      = np.array([])
        # nonlinear constraints at shooting node N
        self.__uh_e    = np.array([])
        self.__lh_e    = np.array([])
        # convex-over-nonlinear constraints
        self.__lphi    = np.array([])
        self.__uphi    = np.array([])
        # nonlinear constraints at shooting node N
        self.__uphi_e = np.array([])
        self.__lphi_e = np.array([])
        # SLACK BOUNDS
        # soft bounds on x
        self.__lsbx   = np.array([])
        self.__usbx   = np.array([])
        self.__idxsbx = np.array([])
        # soft bounds on u
        self.__lsbu   = np.array([])
        self.__usbu   = np.array([])
        self.__idxsbu = np.array([])
        # soft bounds on x at shooting node N
        self.__lsbx_e  = np.array([])
        self.__usbx_e  = np.array([])
        self.__idxsbx_e= np.array([])
        # soft bounds on general linear constraints
        self.__lsg    = np.array([])
        self.__usg    = np.array([])
        self.__idxsg  = np.array([])
        # soft bounds on nonlinear constraints
        self.__lsh    = np.array([])
        self.__ush    = np.array([])
        self.__idxsh  = np.array([])
        # soft bounds on nonlinear constraints
        self.__lsphi  = np.array([])
        self.__usphi  = np.array([])
        self.__idxsphi  = np.array([])
        # soft bounds on general linear constraints at shooting node N
        self.__lsg_e    = np.array([])
        self.__usg_e    = np.array([])
        self.__idxsg_e  = np.array([])
        # soft bounds on nonlinear constraints at shooting node N
        self.__lsh_e    = np.array([])
        self.__ush_e    = np.array([])
        self.__idxsh_e  = np.array([])
        # soft bounds on nonlinear constraints at shooting node N
        self.__lsphi_e    = np.array([])
        self.__usphi_e    = np.array([])
        self.__idxsphi_e  = np.array([])


    # types
    @property
    def constr_type(self):
        """Constraints type for shooting nodes (0 to N-1). string in {BGH, BGP}.
        Default: BGH; BGP is for convex over nonlinear."""
        return self.__constr_type

    @property
    def constr_type_e(self):
        """Constraints type for terminal shooting node N. string in {BGH, BGP}.
        Default: BGH; BGP is for convex over nonlinear."""
        return self.__constr_type_e

    # initial bounds on x
    @property
    def lbx_0(self):
        """:math:`\\underline{x_0}` - lower bounds on x at initial stage 0.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`."""
        return self.__lbx_0

    @property
    def ubx_0(self):
        """:math:`\\bar{x_0}` - upper bounds on x at initial stage 0.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__ubx_0

    @property
    def Jbx_0(self):
        """:math:`J_{bx,0}` - matrix coefficient for bounds on x at initial stage 0.
        Translated internally to :py:attr:`idxbx_0`"""
        print_J_to_idx_note()
        return self.__idxbx_0

    @property
    def idxbx_0(self):
        """Indices of bounds on x at initial stage 0
        -- can be set automatically via x0.
        Can be set by using :py:attr:`Jbx_0`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxbx_0

    @property
    def idxbxe_0(self):
        """Indices of bounds on x0 that are equalities -- can be set automatically via :py:attr:`x0`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxbxe_0

    # bounds on x
    @property
    def lbx(self):
        """:math:`\\underline{x}` - lower bounds on x at intermediate shooting nodes (1 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__lbx

    @property
    def ubx(self):
        """:math:`\\bar{x}` - upper bounds on x at intermediate shooting nodes (1 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__ubx

    @property
    def idxbx(self):
        """indices of bounds on x (defines :math:`J_{bx}`) at intermediate shooting nodes (1 to N-1).
        Can be set by using :py:attr:`Jbx`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxbx

    @property
    def Jbx(self):
        """:math:`J_{bx}` - matrix coefficient for bounds on x
        at intermediate shooting nodes (1 to N-1).
        Translated internally into :py:attr:`idxbx`."""
        print_J_to_idx_note()
        return self.__idxbx

    # bounds on x at shooting node N
    @property
    def lbx_e(self):
        """:math:`\\underline{x}^e` - lower bounds on x at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__lbx_e

    @property
    def ubx_e(self):
        """:math:`\\bar{x}^e` - upper bounds on x at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__ubx_e

    @property
    def idxbx_e(self):
        """Indices for bounds on x at terminal shooting node N (defines :math:`J_{bx}^e`).
        Can be set by using :py:attr:`Jbx_e`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxbx_e

    @property
    def Jbx_e(self):
        """:math:`J_{bx}^e` matrix coefficient for bounds on x at terminal shooting node N.
        Translated internally into :py:attr:`idxbx_e`."""
        print_J_to_idx_note()
        return self.__idxbx_e

    # bounds on u
    @property
    def lbu(self):
        """:math:`\\underline{u}` - lower bounds on u at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`
        """
        return self.__lbu

    @property
    def ubu(self):
        """:math:`\\bar{u}` - upper bounds on u at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`
        """
        return self.__ubu

    @property
    def idxbu(self):
        """Indices of bounds on u (defines :math:`J_{bu}`) at shooting nodes (0 to N-1).
        Can be set by using :py:attr:`Jbu`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`
        """
        return self.__idxbu

    @property
    def Jbu(self):
        """:math:`J_{bu}` - matrix coefficient for bounds on u at shooting nodes (0 to N-1).
        Translated internally to :py:attr:`idxbu`.
        """
        print_J_to_idx_note()
        return self.__idxbu

    # polytopic constraints
    @property
    def C(self):
        """:math:`C` - C matrix in :math:`\\underline{g} \\leq D \, u + C \, x \\leq \\bar{g}`
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array((0,0))`.
        """
        return self.__C

    @property
    def D(self):
        """:math:`D` - D matrix in :math:`\\underline{g} \\leq D \, u + C \, x \\leq \\bar{g}`
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array((0,0))`
        """
        return self.__D

    @property
    def lg(self):
        """:math:`\\underline{g}` - lower bound for general polytopic inequalities
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`
        """
        return self.__lg

    @property
    def ug(self):
        """:math:`\\bar{g}` - upper bound for general polytopic inequalities
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__ug

    # polytopic constraints at shooting node N
    @property
    def C_e(self):
        """:math:`C^e` - C matrix at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array((0,0))`.
        """
        return self.__C_e

    @property
    def lg_e(self):
        """:math:`\\underline{g}^e` - lower bound on general polytopic inequalities
        at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__lg_e

    @property
    def ug_e(self):
        """:math:`\\bar{g}^e` - upper bound on general polytopic inequalities
        at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__ug_e


    # nonlinear constraints
    @property
    def lh(self):
        """:math:`\\underline{h}` - lower bound for nonlinear inequalities
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__lh

    @property
    def uh(self):
        """:math:`\\bar{h}` - upper bound for nonlinear inequalities
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__uh

    # nonlinear constraints at shooting node N
    @property
    def lh_e(self):
        """:math:`\\underline{h}^e` - lower bound on nonlinear inequalities
        at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__lh_e

    @property
    def uh_e(self):
        """:math:`\\bar{h}^e` - upper bound on nonlinear inequalities
        at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__uh_e

    # convex-over-nonlinear constraints
    @property
    def lphi(self):
        """:math:`\\underline{\phi}` - lower bound for convex-over-nonlinear inequalities
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__lphi

    @property
    def uphi(self):
        """:math:`\\bar{\phi}` - upper bound for convex-over-nonlinear inequalities
        at shooting nodes (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__uphi

    # convex-over-nonlinear constraints at shooting node N
    @property
    def lphi_e(self):
        """:math:`\\underline{\phi}^e` - lower bound on convex-over-nonlinear inequalities
        at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__lphi_e

    @property
    def uphi_e(self):
        """:math:`\\bar{\phi}^e` - upper bound on convex-over-nonlinear inequalities
        at terminal shooting node N.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`.
        """
        return self.__uphi_e


    # SLACK bounds
    # soft bounds on x
    @property
    def lsbx(self):
        """Lower bounds on slacks corresponding to soft lower bounds on x
        at stages (1 to N-1);
        not required - zeros by default"""
        return self.__lsbx

    @property
    def usbx(self):
        """Lower bounds on slacks corresponding to soft upper bounds on x
        at stages (1 to N-1);
        not required - zeros by default"""
        return self.__usbx

    @property
    def idxsbx(self):
        """Indices of soft bounds on x within the indices of bounds on x
        at stages (1 to N-1).
        Can be set by using :py:attr:`Jsbx`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxsbx

    @property
    def Jsbx(self):
        """:math:`J_{sbx}` - matrix coefficient for soft bounds on x
        at stages (1 to N-1);
        Translated internally into :py:attr:`idxsbx`."""
        print_J_to_idx_note()
        return self.__idxsbx

    # soft bounds on u
    @property
    def lsbu(self):
        """Lower bounds on slacks corresponding to soft lower bounds on u
        at stages (0 to N-1).
        Not required - zeros by default."""
        return self.__lsbu

    @property
    def usbu(self):
        """Lower bounds on slacks corresponding to soft upper bounds on u
        at stages (0 to N-1);
        not required - zeros by default"""
        return self.__usbu

    @property
    def idxsbu(self):
        """Indices of soft bounds on u within the indices of bounds on u
        at stages (0 to N-1).
        Can be set by using :py:attr:`Jsbu`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxsbu

    @property
    def Jsbu(self):
        """:math:`J_{sbu}` - matrix coefficient for soft bounds on u
        at stages (0 to N-1);
        internally translated into :py:attr:`idxsbu`"""
        print_J_to_idx_note()
        return self.__idxsbu

    # soft bounds on x at shooting node N
    @property
    def lsbx_e(self):
        """Lower bounds on slacks corresponding to soft lower bounds on x at shooting node N.
        Not required - zeros by default"""
        return self.__lsbx_e

    @property
    def usbx_e(self):
        """Lower bounds on slacks corresponding to soft upper bounds on x at shooting node N.
        Not required - zeros by default"""
        return self.__usbx_e

    @property
    def idxsbx_e(self):
        """Indices of soft bounds on x at shooting node N, within the indices of bounds on x at terminal shooting node N.
        Can be set by using :py:attr:`Jsbx_e`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxsbx_e

    @property
    def Jsbx_e(self):
        """:math:`J_{sbx}^e` - matrix coefficient for soft bounds on x at terminal shooting node N.
        Translated internally to :py:attr:`idxsbx_e`"""
        print_J_to_idx_note()
        return self.__idxsbx_e

    # soft general linear constraints
    @property
    def lsg(self):
        """Lower bounds on slacks corresponding to soft lower bounds for general linear constraints
        at stages (0 to N-1).
        Type: :code:`np.ndarray`; default: :code:`np.array([])`
        """
        return self.__lsg

    @property
    def usg(self):
        """Lower bounds on slacks corresponding to soft upper bounds for general linear constraints.
        Not required - zeros by default"""
        return self.__usg

    @property
    def idxsg(self):
        """Indices of soft general linear constraints within the indices of general linear constraints.
        Can be set by using :py:attr:`Jsg`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxsg

    @property
    def Jsg(self):
        """:math:`J_{sg}` - matrix coefficient for soft bounds on general linear constraints.
        Translated internally to :py:attr:`idxsg`"""
        print_J_to_idx_note()
        return self.__idxsg

    # soft nonlinear constraints
    @property
    def lsh(self):
        """Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints.
        Not required - zeros by default"""
        return self.__lsh

    @property
    def ush(self):
        """Lower bounds on slacks corresponding to soft upper bounds for nonlinear constraints.
        Not required - zeros by default"""
        return self.__ush

    @property
    def idxsh(self):
        """Indices of soft nonlinear constraints within the indices of nonlinear constraints.
        Can be set by using :py:attr:`Jbx`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxsh

    @property
    def Jsh(self):
        """:math:`J_{sh}` - matrix coefficient for soft bounds on nonlinear constraints.
        Translated internally to :py:attr:`idxsh`"""
        print_J_to_idx_note()
        return self.__idxsh

    # soft bounds on convex-over-nonlinear constraints
    @property
    def lsphi(self):
        """Lower bounds on slacks corresponding to soft lower bounds for convex-over-nonlinear constraints.
        Not required - zeros by default"""
        return self.__lsphi

    @property
    def usphi(self):
        """Lower bounds on slacks corresponding to soft upper bounds for convex-over-nonlinear constraints.
        Not required - zeros by default"""
        return self.__usphi

    @property
    def idxsphi(self):
        """Indices of soft convex-over-nonlinear constraints within the indices of nonlinear constraints.
        Can be set by using :py:attr:`Jsphi`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxsphi

    @property
    def Jsphi(self):
        """:math:`J_{s, \phi}` - matrix coefficient for soft bounds on convex-over-nonlinear constraints.
        Translated internally into :py:attr:`idxsphi`."""
        print_J_to_idx_note()
        return self.__idxsphi


    # soft bounds on general linear constraints at shooting node N
    @property
    def lsg_e(self):
        """Lower bounds on slacks corresponding to soft lower bounds for general linear constraints at shooting node N.
        Not required - zeros by default"""
        return self.__lsg_e

    @property
    def usg_e(self):
        """Lower bounds on slacks corresponding to soft upper bounds for general linear constraints at shooting node N.
        Not required - zeros by default"""
        return self.__usg_e

    @property
    def idxsg_e(self):
        """Indices of soft general linear constraints at shooting node N within the indices of general linear constraints at shooting node N.
        Can be set by using :py:attr:`Jsg_e`."""
        return self.__idxsg_e

    @property
    def Jsg_e(self):
        """:math:`J_{s,h}^e` - matrix coefficient for soft bounds on general linear constraints at terminal shooting node N.
        Translated internally to :py:attr:`idxsg_e`"""
        print_J_to_idx_note()
        return self.__idxsg_e


    # soft bounds on nonlinear constraints at shooting node N
    @property
    def lsh_e(self):
        """Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints at terminal shooting node N.
        Not required - zeros by default"""
        return self.__lsh_e

    @property
    def ush_e(self):
        """Lower bounds on slacks corresponding to soft upper bounds for nonlinear constraints at terminal shooting node N.
        Not required - zeros by default"""
        return self.__ush_e

    @property
    def idxsh_e(self):
        """Indices of soft nonlinear constraints at shooting node N within the indices of nonlinear constraints at terminal shooting node N.
        Can be set by using :py:attr:`Jsh_e`."""
        return self.__idxsh_e

    @property
    def Jsh_e(self):
        """:math:`J_{s,h}^e` - matrix coefficient for soft bounds on nonlinear constraints at terminal shooting node N; fills :py:attr:`idxsh_e`"""
        print_J_to_idx_note()
        return self.__idxsh_e

    # soft bounds on convex-over-nonlinear constraints at shooting node N
    @property
    def lsphi_e(self):
        """Lower bounds on slacks corresponding to soft lower bounds for convex-over-nonlinear constraints at terminal shooting node N.
        Not required - zeros by default"""
        return self.__lsphi_e

    @property
    def usphi_e(self):
        """Lower bounds on slacks corresponding to soft upper bounds for convex-over-nonlinear constraints at terminal shooting node N.
        Not required - zeros by default"""
        return self.__usphi_e

    @property
    def idxsphi_e(self):
        """Indices of soft nonlinear constraints at shooting node N within the indices of nonlinear constraints at terminal shooting node N.
        Can be set by using :py:attr:`Jsphi_e`.
        Type: :code:`np.ndarray`; default: :code:`np.array([])`"""
        return self.__idxsphi_e

    @property
    def Jsphi_e(self):
        """:math:`J_{sh}^e` - matrix coefficient for soft bounds on convex-over-nonlinear constraints at shooting node N.
        Translated internally to :py:attr:`idxsphi_e`"""
        print_J_to_idx_note()
        return self.__idxsphi_e

    @property
    def x0(self):
        """:math:`x_0 \\in \mathbb{R}^{n_x}` - initial state --
        Translated internally to :py:attr:`idxbx_0`, :py:attr:`lbx_0`, :py:attr:`ubx_0`, :py:attr:`idxbxe_0` """
        print("x0 is converted to lbx_0, ubx_0, idxbx_0")
        print("idxbx_0: ", self.__idxbx_0)
        print("lbx_0: ", self.__lbx_0)
        print("ubx_0: ", self.__ubx_0)
        print("idxbxe_0: ", self.__idxbxe_0)
        return None

    # SETTERS
    @constr_type.setter
    def constr_type(self, constr_type):
        constr_types = ('BGH', 'BGP')
        if constr_type in constr_types:
            self.__constr_type = constr_type
        else:
            raise Exception('Invalid constr_type value. Possible values are:\n\n' \
                    + ',\n'.join(constr_types) + '.\n\nYou have: ' + constr_type + '.\n\n')

    @constr_type_e.setter
    def constr_type_e(self, constr_type_e):
        constr_types = ('BGH', 'BGP')
        if constr_type_e in constr_types:
            self.__constr_type_e = constr_type_e
        else:
            raise Exception('Invalid constr_type_e value. Possible values are:\n\n' \
                    + ',\n'.join(constr_types) + '.\n\nYou have: ' + constr_type_e + '.\n\n')

    # initial x
    @lbx_0.setter
    def lbx_0(self, lbx_0):
        if isinstance(lbx_0, np.ndarray):
            self.__lbx_0 = lbx_0
        else:
            raise Exception('Invalid lbx_0 value.')

    @ubx_0.setter
    def ubx_0(self, ubx_0):
        if isinstance(ubx_0, np.ndarray):
            self.__ubx_0 = ubx_0
        else:
            raise Exception('Invalid ubx_0 value.')

    @idxbx_0.setter
    def idxbx_0(self, idxbx_0):
        if isinstance(idxbx_0, np.ndarray):
            self.__idxbx_0 = idxbx_0
        else:
            raise Exception('Invalid idxbx_0 value.')

    @Jbx_0.setter
    def Jbx_0(self, Jbx_0):
        if isinstance(Jbx_0, np.ndarray):
            self.__idxbx_0 = J_to_idx(Jbx_0)
        else:
            raise Exception('Invalid Jbx_0 value.')

    @idxbxe_0.setter
    def idxbxe_0(self, idxbxe_0):
        if isinstance(idxbxe_0, np.ndarray):
            self.__idxbxe_0 = idxbxe_0
        else:
            raise Exception('Invalid idxbxe_0 value.')


    @x0.setter
    def x0(self, x0):
        if isinstance(x0, np.ndarray):
            self.__lbx_0 = x0
            self.__ubx_0 = x0
            self.__idxbx_0 = np.arange(x0.size)
            self.__idxbxe_0 = np.arange(x0.size)
        else:
            raise Exception('Invalid x0 value.')

    # bounds on x
    @lbx.setter
    def lbx(self, lbx):
        if isinstance(lbx, np.ndarray):
            self.__lbx = lbx
        else:
            raise Exception('Invalid lbx value.')

    @ubx.setter
    def ubx(self, ubx):
        if isinstance(ubx, np.ndarray):
            self.__ubx = ubx
        else:
            raise Exception('Invalid ubx value.')

    @idxbx.setter
    def idxbx(self, idxbx):
        if isinstance(idxbx, np.ndarray):
            self.__idxbx = idxbx
        else:
            raise Exception('Invalid idxbx value.')

    @Jbx.setter
    def Jbx(self, Jbx):
        if isinstance(Jbx, np.ndarray):
            self.__idxbx = J_to_idx(Jbx)
        else:
            raise Exception('Invalid Jbx value.')

    # bounds on u
    @lbu.setter
    def lbu(self, lbu):
        if isinstance(lbu, np.ndarray):
            self.__lbu = lbu
        else:
            raise Exception('Invalid lbu value.')

    @ubu.setter
    def ubu(self, ubu):
        if isinstance(ubu, np.ndarray):
            self.__ubu = ubu
        else:
            raise Exception('Invalid ubu value.')

    @idxbu.setter
    def idxbu(self, idxbu):
        if isinstance(idxbu, np.ndarray):
            self.__idxbu = idxbu
        else:
            raise Exception('Invalid idxbu value.')

    @Jbu.setter
    def Jbu(self, Jbu):
        if isinstance(Jbu, np.ndarray):
            self.__idxbu = J_to_idx(Jbu)
        else:
            raise Exception('Invalid Jbu value.')

    # bounds on x at shooting node N
    @lbx_e.setter
    def lbx_e(self, lbx_e):
        if isinstance(lbx_e, np.ndarray):
            self.__lbx_e = lbx_e
        else:
            raise Exception('Invalid lbx_e value.')

    @ubx_e.setter
    def ubx_e(self, ubx_e):
        if isinstance(ubx_e, np.ndarray):
            self.__ubx_e = ubx_e
        else:
            raise Exception('Invalid ubx_e value.')

    @idxbx_e.setter
    def idxbx_e(self, idxbx_e):
        if isinstance(idxbx_e, np.ndarray):
            self.__idxbx_e = idxbx_e
        else:
            raise Exception('Invalid idxbx_e value.')

    @Jbx_e.setter
    def Jbx_e(self, Jbx_e):
        if isinstance(Jbx_e, np.ndarray):
            self.__idxbx_e = J_to_idx(Jbx_e)
        else:
            raise Exception('Invalid Jbx_e value.')

    # polytopic constraints
    @D.setter
    def D(self, D):
        if isinstance(D, np.ndarray) and len(D.shape) == 2:
            self.__D = D
        else:
            raise Exception('Invalid constraint D value.' \
                + 'Should be 2 dimensional numpy array.')

    @C.setter
    def C(self, C):
        if isinstance(C, np.ndarray) and len(C.shape) == 2:
            self.__C = C
        else:
            raise Exception('Invalid constraint C value.' \
                + 'Should be 2 dimensional numpy array.')

    @lg.setter
    def lg(self, lg):
        if isinstance(lg, np.ndarray):
            self.__lg = lg
        else:
            raise Exception('Invalid lg value.')

    @ug.setter
    def ug(self, ug):
        if isinstance(ug, np.ndarray):
            self.__ug = ug
        else:
            raise Exception('Invalid ug value.')

    # polytopic constraints at shooting node N
    @C_e.setter
    def C_e(self, C_e):
        if isinstance(C_e, np.ndarray) and len(C_e.shape) == 2:
            self.__C_e = C_e
        else:
            raise Exception('Invalid constraint C_e value.' \
                + 'Should be 2 dimensional numpy array.')

    @lg_e.setter
    def lg_e(self, lg_e):
        if isinstance(lg_e, np.ndarray):
            self.__lg_e = lg_e
        else:
            raise Exception('Invalid lg_e value.')

    @ug_e.setter
    def ug_e(self, ug_e):
        if isinstance(ug_e, np.ndarray):
            self.__ug_e = ug_e
        else:
            raise Exception('Invalid ug_e value.')

    # nonlinear constraints
    @lh.setter
    def lh(self, lh):
        if isinstance(lh, np.ndarray):
            self.__lh = lh
        else:
            raise Exception('Invalid lh value.')

    @uh.setter
    def uh(self, uh):
        if isinstance(uh, np.ndarray):
            self.__uh = uh
        else:
            raise Exception('Invalid uh value.')

    # convex-over-nonlinear constraints
    @lphi.setter
    def lphi(self, lphi):
        if isinstance(lphi, np.ndarray):
            self.__lphi = lphi
        else:
            raise Exception('Invalid lphi value.')

    @uphi.setter
    def uphi(self, uphi):
        if isinstance(uphi, np.ndarray):
            self.__uphi = uphi
        else:
            raise Exception('Invalid uphi value.')

    # nonlinear constraints at shooting node N
    @lh_e.setter
    def lh_e(self, lh_e):
        if isinstance(lh_e, np.ndarray):
            self.__lh_e = lh_e
        else:
            raise Exception('Invalid lh_e value.')

    @uh_e.setter
    def uh_e(self, uh_e):
        if isinstance(uh_e, np.ndarray):
            self.__uh_e = uh_e
        else:
            raise Exception('Invalid uh_e value.')

    # convex-over-nonlinear constraints at shooting node N
    @lphi_e.setter
    def lphi_e(self, lphi_e):
        if isinstance(lphi_e, np.ndarray):
            self.__lphi_e = lphi_e
        else:
            raise Exception('Invalid lphi_e value.')

    @uphi_e.setter
    def uphi_e(self, uphi_e):
        if isinstance(uphi_e, np.ndarray):
            self.__uphi_e = uphi_e
        else:
            raise Exception('Invalid uphi_e value.')

    # SLACK bounds
    # soft bounds on x
    @lsbx.setter
    def lsbx(self, lsbx):
        if isinstance(lsbx, np.ndarray):
            self.__lsbx = lsbx
        else:
            raise Exception('Invalid lsbx value.')

    @usbx.setter
    def usbx(self, usbx):
        if isinstance(usbx, np.ndarray):
            self.__usbx = usbx
        else:
            raise Exception('Invalid usbx value.')

    @idxsbx.setter
    def idxsbx(self, idxsbx):
        if isinstance(idxsbx, np.ndarray):
            self.__idxsbx = idxsbx
        else:
            raise Exception('Invalid idxsbx value.')

    @Jsbx.setter
    def Jsbx(self, Jsbx):
        if isinstance(Jsbx, np.ndarray):
            self.__idxsbx = J_to_idx_slack(Jsbx)
        else:
            raise Exception('Invalid Jsbx value, expected numpy array.')

    # soft bounds on u
    @lsbu.setter
    def lsbu(self, lsbu):
        if isinstance(lsbu, np.ndarray):
            self.__lsbu = lsbu
        else:
            raise Exception('Invalid lsbu value.')

    @usbu.setter
    def usbu(self, usbu):
        if isinstance(usbu, np.ndarray):
            self.__usbu = usbu
        else:
            raise Exception('Invalid usbu value.')

    @idxsbu.setter
    def idxsbu(self, idxsbu):
        if isinstance(idxsbu, np.ndarray):
            self.__idxsbu = idxsbu
        else:
            raise Exception('Invalid idxsbu value.')

    @Jsbu.setter
    def Jsbu(self, Jsbu):
        if isinstance(Jsbu, np.ndarray):
            self.__idxsbu = J_to_idx_slack(Jsbu)
        else:
            raise Exception('Invalid Jsbu value.')

    # soft bounds on x at shooting node N
    @lsbx_e.setter
    def lsbx_e(self, lsbx_e):
        if isinstance(lsbx_e, np.ndarray):
            self.__lsbx_e = lsbx_e
        else:
            raise Exception('Invalid lsbx_e value.')

    @usbx_e.setter
    def usbx_e(self, usbx_e):
        if isinstance(usbx_e, np.ndarray):
            self.__usbx_e = usbx_e
        else:
            raise Exception('Invalid usbx_e value.')

    @idxsbx_e.setter
    def idxsbx_e(self, idxsbx_e):
        if isinstance(idxsbx_e, np.ndarray):
            self.__idxsbx_e = idxsbx_e
        else:
            raise Exception('Invalid idxsbx_e value.')

    @Jsbx_e.setter
    def Jsbx_e(self, Jsbx_e):
        if isinstance(Jsbx_e, np.ndarray):
            self.__idxsbx_e = J_to_idx_slack(Jsbx_e)
        else:
            raise Exception('Invalid Jsbx_e value.')


    # soft bounds on general linear constraints
    @lsg.setter
    def lsg(self, lsg):
        if isinstance(lsg, np.ndarray):
            self.__lsg = lsg
        else:
            raise Exception('Invalid lsg value.')

    @usg.setter
    def usg(self, usg):
        if isinstance(usg, np.ndarray):
            self.__usg = usg
        else:
            raise Exception('Invalid usg value.')

    @idxsg.setter
    def idxsg(self, idxsg):
        if isinstance(idxsg, np.ndarray):
            self.__idxsg = idxsg
        else:
            raise Exception('Invalid idxsg value.')

    @Jsg.setter
    def Jsg(self, Jsg):
        if isinstance(Jsg, np.ndarray):
            self.__idxsg = J_to_idx_slack(Jsg)
        else:
            raise Exception('Invalid Jsg value, expected numpy array.')


    # soft bounds on nonlinear constraints
    @lsh.setter
    def lsh(self, lsh):
        if isinstance(lsh, np.ndarray):
            self.__lsh = lsh
        else:
            raise Exception('Invalid lsh value.')

    @ush.setter
    def ush(self, ush):
        if isinstance(ush, np.ndarray):
            self.__ush = ush
        else:
            raise Exception('Invalid ush value.')

    @idxsh.setter
    def idxsh(self, idxsh):
        if isinstance(idxsh, np.ndarray):
            self.__idxsh = idxsh
        else:
            raise Exception('Invalid idxsh value.')


    @Jsh.setter
    def Jsh(self, Jsh):
        if isinstance(Jsh, np.ndarray):
            self.__idxsh = J_to_idx_slack(Jsh)
        else:
            raise Exception('Invalid Jsh value, expected numpy array.')

    # soft bounds on convex-over-nonlinear constraints
    @lsphi.setter
    def lsphi(self, lsphi):
        if isinstance(lsphi, np.ndarray):
            self.__lsphi = lsphi
        else:
            raise Exception('Invalid lsphi value.')

    @usphi.setter
    def usphi(self, usphi):
        if isinstance(usphi, np.ndarray):
            self.__usphi = usphi
        else:
            raise Exception('Invalid usphi value.')

    @idxsphi.setter
    def idxsphi(self, idxsphi):
        if isinstance(idxsphi, np.ndarray):
            self.__idxsphi = idxsphi
        else:
            raise Exception('Invalid idxsphi value.')

    @Jsphi.setter
    def Jsphi(self, Jsphi):
        if isinstance(Jsphi, np.ndarray):
            self.__idxsphi = J_to_idx_slack(Jsphi)
        else:
            raise Exception('Invalid Jsphi value, expected numpy array.')

    # soft bounds on general linear constraints at shooting node N
    @lsg_e.setter
    def lsg_e(self, lsg_e):
        if isinstance(lsg_e, np.ndarray):
            self.__lsg_e = lsg_e
        else:
            raise Exception('Invalid lsg_e value.')

    @usg_e.setter
    def usg_e(self, usg_e):
        if isinstance(usg_e, np.ndarray):
            self.__usg_e = usg_e
        else:
            raise Exception('Invalid usg_e value.')

    @idxsg_e.setter
    def idxsg_e(self, idxsg_e):
        if isinstance(idxsg_e, np.ndarray):
            self.__idxsg_e = idxsg_e
        else:
            raise Exception('Invalid idxsg_e value.')

    @Jsg_e.setter
    def Jsg_e(self, Jsg_e):
        if isinstance(Jsg_e, np.ndarray):
            self.__idxsg_e = J_to_idx_slack(Jsg_e)
        else:
            raise Exception('Invalid Jsg_e value, expected numpy array.')

    # soft bounds on nonlinear constraints at shooting node N
    @lsh_e.setter
    def lsh_e(self, lsh_e):
        if isinstance(lsh_e, np.ndarray):
            self.__lsh_e = lsh_e
        else:
            raise Exception('Invalid lsh_e value.')

    @ush_e.setter
    def ush_e(self, ush_e):
        if isinstance(ush_e, np.ndarray):
            self.__ush_e = ush_e
        else:
            raise Exception('Invalid ush_e value.')

    @idxsh_e.setter
    def idxsh_e(self, idxsh_e):
        if isinstance(idxsh_e, np.ndarray):
            self.__idxsh_e = idxsh_e
        else:
            raise Exception('Invalid idxsh_e value.')

    @Jsh_e.setter
    def Jsh_e(self, Jsh_e):
        if isinstance(Jsh_e, np.ndarray):
            self.__idxsh_e = J_to_idx_slack(Jsh_e)
        else:
            raise Exception('Invalid Jsh_e value, expected numpy array.')


    # soft bounds on convex-over-nonlinear constraints at shooting node N
    @lsphi_e.setter
    def lsphi_e(self, lsphi_e):
        if isinstance(lsphi_e, np.ndarray):
            self.__lsphi_e = lsphi_e
        else:
            raise Exception('Invalid lsphi_e value.')

    @usphi_e.setter
    def usphi_e(self, usphi_e):
        if isinstance(usphi_e, np.ndarray):
            self.__usphi_e = usphi_e
        else:
            raise Exception('Invalid usphi_e value.')

    @idxsphi_e.setter
    def idxsphi_e(self, idxsphi_e):
        if isinstance(idxsphi_e, np.ndarray):
            self.__idxsphi_e = idxsphi_e
        else:
            raise Exception('Invalid idxsphi_e value.')

    @Jsphi_e.setter
    def Jsphi_e(self, Jsphi_e):
        if isinstance(Jsphi_e, np.ndarray):
            self.__idxsphi_e = J_to_idx_slack(Jsphi_e)
        else:
            raise Exception('Invalid Jsphi_e value.')

    def set(self, attr, value):
        setattr(self, attr, value)


class AcadosOcpOptions:
    """
    class containing the description of the solver options
    """
    def __init__(self):
        self.__qp_solver        = 'PARTIAL_CONDENSING_HPIPM'  # qp solver to be used in the NLP solver
        self.__hessian_approx   = 'GAUSS_NEWTON'              # hessian approximation
        self.__integrator_type  = 'ERK'                       # integrator type
        self.__tf               = None                        # prediction horizon
        self.__nlp_solver_type  = 'SQP_RTI'                   # NLP solver
        self.__globalization = 'FIXED_STEP'
        self.__nlp_solver_step_length = 1.0                   # fixed Newton step length
        self.__levenberg_marquardt = 0.0
        self.__collocation_type = 'GAUSS_LEGENDRE'
        self.__sim_method_num_stages  = 4                     # number of stages in the integrator
        self.__sim_method_num_steps   = 1                     # number of steps in the integrator
        self.__sim_method_newton_iter = 3                     # number of Newton iterations in simulation method
        self.__sim_method_newton_tol = 0.0
        self.__sim_method_jac_reuse = 0
        self.__qp_solver_tol_stat = None                      # QP solver stationarity tolerance
        self.__qp_solver_tol_eq   = None                      # QP solver equality tolerance
        self.__qp_solver_tol_ineq = None                      # QP solver inequality
        self.__qp_solver_tol_comp = None                      # QP solver complementarity
        self.__qp_solver_iter_max = 50                        # QP solver max iter
        self.__qp_solver_cond_N = None                        # QP solver: new horizon after partial condensing
        self.__qp_solver_warm_start = 0
        self.__qp_solver_cond_ric_alg = 1
        self.__qp_solver_ric_alg = 1
        self.__nlp_solver_tol_stat = 1e-6                     # NLP solver stationarity tolerance
        self.__nlp_solver_tol_eq   = 1e-6                     # NLP solver equality tolerance
        self.__nlp_solver_tol_ineq = 1e-6                     # NLP solver inequality
        self.__nlp_solver_tol_comp = 1e-6                     # NLP solver complementarity
        self.__nlp_solver_max_iter = 100                      # NLP solver maximum number of iterations
        self.__nlp_solver_ext_qp_res = 0
        self.__Tsim = None                                    # automatically calculated as tf/N
        self.__print_level = 0                                # print level
        self.__initialize_t_slacks = 0                        # possible values: 0, 1
        self.__regularize_method = None
        self.__time_steps = None
        self.__shooting_nodes = None
        self.__exact_hess_cost = 1
        self.__exact_hess_dyn = 1
        self.__exact_hess_constr = 1
        self.__ext_cost_num_hess = 0
        self.__alpha_min = 0.05
        self.__alpha_reduction = 0.7
        self.__line_search_use_sufficient_descent = 0
        self.__globalization_use_SOC = 0
        self.__full_step_dual = 0
        self.__eps_sufficient_descent = 1e-4
        self.__hpipm_mode = 'BALANCE'
        # TODO: move those out? they are more about generation than about the acados OCP solver.
        self.__ext_fun_compile_flags = '-O2'
        self.__model_external_shared_lib_dir   = None         # path to the the .so lib
        self.__model_external_shared_lib_name  = None         # name of the the .so lib
        self.__custom_update_filename = ''
        self.__custom_update_header_filename = ''
        self.__custom_templates = []
        self.__custom_update_copy = True

    @property
    def qp_solver(self):
        """QP solver to be used in the NLP solver.
        String in ('PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP').
        Default: 'PARTIAL_CONDENSING_HPIPM'.
        """
        return self.__qp_solver

    @property
    def ext_fun_compile_flags(self):
        """
        String with compiler flags for external function compilation.
        Default: '-O2'.
        """
        return self.__ext_fun_compile_flags


    @property
    def custom_update_filename(self):
        """
        Filename of the custom C function to update solver data and parameters in between solver calls

        This file has to implement the functions
        int custom_update_init_function([model.name]_solver_capsule* capsule);
        int custom_update_function([model.name]_solver_capsule* capsule);
        int custom_update_terminate_function([model.name]_solver_capsule* capsule);


        Default: ''.
        """
        return self.__custom_update_filename


    @property
    def custom_templates(self):
        """
        List of tuples of the form:
        (input_filename, output_filename)

        Custom templates are render in OCP solver generation.

        Default: [].
        """
        return self.__custom_templates


    @property
    def custom_update_header_filename(self):
        """
        Header filename of the custom C function to update solver data and parameters in between solver calls

        This file has to declare the custom_update functions and look as follows:

        ```
        // Called at the end of solver creation.
        // This is allowed to allocate memory and store the pointer to it into capsule->custom_update_memory.
        int custom_update_init_function([model.name]_solver_capsule* capsule);

        // Custom update function that can be called between solver calls
        int custom_update_function([model.name]_solver_capsule* capsule, double* data, int data_len);

        // Called just before destroying the solver.
        // Responsible to free allocated memory, stored at capsule->custom_update_memory.
        int custom_update_terminate_function([model.name]_solver_capsule* capsule);

        Default: ''.
        """
        return self.__custom_update_header_filename

    @property
    def custom_update_copy(self):
        """
        Boolean;
        If True, the custom update function files are copied into the `code_export_directory`.
        """
        return self.__custom_update_copy


    @property
    def hpipm_mode(self):
        """
        Mode of HPIPM to be used,

        String in ('BALANCE', 'SPEED_ABS', 'SPEED', 'ROBUST').

        Default: 'BALANCE'.

        see https://cdn.syscop.de/publications/Frison2020a.pdf
        and the HPIPM code:
        https://github.com/giaf/hpipm/blob/master/ocp_qp/x_ocp_qp_ipm.c#L69
        """
        return self.__hpipm_mode

    @property
    def hessian_approx(self):
        """Hessian approximation.
        String in ('GAUSS_NEWTON', 'EXACT').
        Default: 'GAUSS_NEWTON'.
        """
        return self.__hessian_approx

    @property
    def integrator_type(self):
        """
        Integrator type.
        String in ('ERK', 'IRK', 'GNSF', 'DISCRETE', 'LIFTED_IRK').
        Default: 'ERK'.
        """
        return self.__integrator_type

    @property
    def nlp_solver_type(self):
        """NLP solver.
        String in ('SQP', 'SQP_RTI').
        Default: 'SQP_RTI'.
        """
        return self.__nlp_solver_type

    @property
    def globalization(self):
        """Globalization type.
        String in ('FIXED_STEP', 'MERIT_BACKTRACKING').
        Default: 'FIXED_STEP'.

        .. note:: preliminary implementation.
        """
        return self.__globalization

    @property
    def collocation_type(self):
        """Collocation type: relevant for implicit integrators
        -- string in {GAUSS_RADAU_IIA, GAUSS_LEGENDRE}.

        Default: GAUSS_LEGENDRE
        """
        return self.__collocation_type

    @property
    def regularize_method(self):
        """Regularization method for the Hessian.
        String in ('NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS', 'CONVEXIFY') or :code:`None`.

        - MIRROR: performs eigenvalue decomposition H = V^T D V and sets D_ii = max(eps, abs(D_ii))
        - PROJECT: performs eigenvalue decomposition H = V^T D V and sets D_ii = max(eps, D_ii)
        - CONVEXIFY: Algorithm 6 from Verschueren2017, https://cdn.syscop.de/publications/Verschueren2017.pdf
        - PROJECT_REDUC_HESS: experimental

        Note: default eps = 1e-4

        Default: :code:`None`.
        """
        return self.__regularize_method

    @property
    def nlp_solver_step_length(self):
        """
        Fixed Newton step length.
        Type: float > 0.
        Default: 1.0.
        """
        return self.__nlp_solver_step_length

    @property
    def levenberg_marquardt(self):
        """
        Factor for LM regularization.
        Type: float >= 0
        Default: 0.0.
        """
        return self.__levenberg_marquardt

    @property
    def sim_method_num_stages(self):
        """
        Number of stages in the integrator.
        Type: int > 0 or ndarray of ints > 0 of shape (N,).
        Default: 4
        """
        return self.__sim_method_num_stages

    @property
    def sim_method_num_steps(self):
        """
        Number of steps in the integrator.
        Type: int > 0 or ndarray of ints > 0 of shape (N,).
        Default: 1
        """
        return self.__sim_method_num_steps

    @property
    def sim_method_newton_iter(self):
        """
        Number of Newton iterations in simulation method.
        Type: int > 0
        Default: 3
        """
        return self.__sim_method_newton_iter

    @property
    def sim_method_newton_tol(self):
        """
        Tolerance of Newton system in simulation method.
        Type: float: 0.0 means not used
        Default: 0.0
        """
        return self.__sim_method_newton_tol

    @property
    def sim_method_jac_reuse(self):
        """
        Integer determining if jacobians are reused within integrator or ndarray of ints > 0 of shape (N,).
        0: False (no reuse); 1: True (reuse)
        Default: 0
        """
        return self.__sim_method_jac_reuse

    @property
    def qp_solver_tol_stat(self):
        """
        QP solver stationarity tolerance.
        Default: :code:`None`
        """
        return self.__qp_solver_tol_stat

    @property
    def qp_solver_tol_eq(self):
        """
        QP solver equality tolerance.
        Default: :code:`None`
        """
        return self.__qp_solver_tol_eq

    @property
    def qp_solver_tol_ineq(self):
        """
        QP solver inequality.
        Default: :code:`None`
        """
        return self.__qp_solver_tol_ineq

    @property
    def qp_solver_tol_comp(self):
        """
        QP solver complementarity.
        Default: :code:`None`
        """
        return self.__qp_solver_tol_comp

    @property
    def qp_solver_cond_N(self):
        """QP solver: New horizon after partial condensing.
        Set to N by default -> no condensing."""
        return self.__qp_solver_cond_N

    @property
    def qp_solver_warm_start(self):
        """
        QP solver: Warm starting.
        0: no warm start; 1: warm start; 2: hot start.
        Default: 0
        """
        return self.__qp_solver_warm_start

    @property
    def qp_solver_cond_ric_alg(self):
        """
        QP solver: Determines which algorithm is used in HPIPM condensing.
        0: dont factorize hessian in the condensing; 1: factorize.
        Default: 1
        """
        return self.__qp_solver_cond_ric_alg

    @property
    def qp_solver_ric_alg(self):
        """
        QP solver: Determines which algorithm is used in HPIPM OCP QP solver.
        0 classical Riccati, 1 square-root Riccati.

        Note: taken from [HPIPM paper]:

        (a) the classical implementation requires the reduced Hessian with respect to the dynamics
            equality constraints to be positive definite, but allows the full-space Hessian to be indefinite)
        (b) the square-root implementation, which in order to reduce the flop count employs the Cholesky
            factorization of the Riccati recursion matrix, and therefore requires the full-space Hessian to be positive definite

        [HPIPM paper]: HPIPM: a high-performance quadratic programming framework for model predictive control, Frison and Diehl, 2020
        https://cdn.syscop.de/publications/Frison2020a.pdf

        Default: 1
        """
        return self.__qp_solver_ric_alg

    @property
    def qp_solver_iter_max(self):
        """
        QP solver: maximum number of iterations.
        Type: int > 0
        Default: 50
        """
        return self.__qp_solver_iter_max

    @property
    def tol(self):
        """
        NLP solver tolerance. Sets or gets the max of :py:attr:`nlp_solver_tol_eq`,
        :py:attr:`nlp_solver_tol_ineq`, :py:attr:`nlp_solver_tol_comp`
        and :py:attr:`nlp_solver_tol_stat`.
        """
        return max([self.__nlp_solver_tol_eq, self.__nlp_solver_tol_ineq,\
                    self.__nlp_solver_tol_comp, self.__nlp_solver_tol_stat])

    @property
    def qp_tol(self):
        """
        QP solver tolerance.
        Sets all of the following at once or gets the max of
        :py:attr:`qp_solver_tol_eq`, :py:attr:`qp_solver_tol_ineq`,
        :py:attr:`qp_solver_tol_comp` and
        :py:attr:`qp_solver_tol_stat`.
        """
        return max([self.__qp_solver_tol_eq, self.__qp_solver_tol_ineq,\
                    self.__qp_solver_tol_comp, self.__qp_solver_tol_stat])

    @property
    def nlp_solver_tol_stat(self):
        """
        NLP solver stationarity tolerance.
        Type: float > 0
        Default: 1e-6
        """
        return self.__nlp_solver_tol_stat

    @property
    def nlp_solver_tol_eq(self):
        """NLP solver equality tolerance"""
        return self.__nlp_solver_tol_eq

    @property
    def alpha_min(self):
        """Minimal step size for globalization MERIT_BACKTRACKING, default: 0.05."""
        return self.__alpha_min

    @property
    def alpha_reduction(self):
        """Step size reduction factor for globalization MERIT_BACKTRACKING, default: 0.7."""
        return self.__alpha_reduction

    @property
    def line_search_use_sufficient_descent(self):
        """
        Determines if sufficient descent (Armijo) condition is used in line search.
        Type: int; 0 or 1;
        default: 0.
        """
        return self.__line_search_use_sufficient_descent

    @property
    def eps_sufficient_descent(self):
        """
        Factor for sufficient descent (Armijo) conditon, see line_search_use_sufficient_descent.
        Type: float,
        default: 1e-4.
        """
        return self.__eps_sufficient_descent

    @property
    def globalization_use_SOC(self):
        """
        Determines if second order correction (SOC) is done when using MERIT_BACKTRACKING.
        SOC is done if preliminary line search does not return full step.
        Type: int; 0 or 1;
        default: 0.
        """
        return self.__globalization_use_SOC

    @property
    def full_step_dual(self):
        """
        Determines if dual variables are updated with full steps (alpha=1.0) when primal variables are updated with smaller step.
        Type: int; 0 or 1;
        default: 0.
        """
        return self.__full_step_dual

    @property
    def nlp_solver_tol_ineq(self):
        """NLP solver inequality tolerance"""
        return self.__nlp_solver_tol_ineq

    @property
    def nlp_solver_ext_qp_res(self):
        """Determines if residuals of QP are computed externally within NLP solver (for debugging)

        Type: int; 0 or 1;
        Default: 0.
        """
        return self.__nlp_solver_ext_qp_res

    @property
    def nlp_solver_tol_comp(self):
        """NLP solver complementarity tolerance"""
        return self.__nlp_solver_tol_comp

    @property
    def nlp_solver_max_iter(self):
        """
        NLP solver maximum number of iterations.
        Type: int > 0
        Default: 100
        """
        return self.__nlp_solver_max_iter

    @property
    def time_steps(self):
        """
        Vector with time steps between the shooting nodes. Set automatically to uniform discretization if :py:attr:`N` and :py:attr:`tf` are provided.
        Default: :code:`None`
        """
        return self.__time_steps

    @property
    def shooting_nodes(self):
        """
        Vector with the shooting nodes, time_steps will be computed from it automatically.
        Default: :code:`None`
        """
        return self.__shooting_nodes

    @property
    def tf(self):
        """
        Prediction horizon
        Type: float > 0
        Default: :code:`None`
        """
        return self.__tf

    @property
    def Tsim(self):
        """
        Time horizon for one integrator step. Automatically calculated as :py:attr:`tf`/:py:attr:`N`.
        Default: :code:`None`
        """
        return self.__Tsim

    @property
    def print_level(self):
        """
        Verbosity of printing.
        Type: int >= 0
        Default: 0
        """
        return self.__print_level

    @property
    def model_external_shared_lib_dir(self):
        """Path to the .so lib"""
        return self.__model_external_shared_lib_dir

    @property
    def model_external_shared_lib_name(self):
        """Name of the .so lib"""
        return self.__model_external_shared_lib_name

    @property
    def exact_hess_constr(self):
        """
        Used in case of hessian_approx == 'EXACT'.\n
        Can be used to turn off exact hessian contributions from the constraints module.
        """
        return self.__exact_hess_constr

    @property
    def exact_hess_cost(self):
        """
        Used in case of hessian_approx == 'EXACT'.\n
        Can be used to turn off exact hessian contributions from the cost module.
        """
        return self.__exact_hess_cost

    @property
    def exact_hess_dyn(self):
        """
        Used in case of hessian_approx == 'EXACT'.\n
        Can be used to turn off exact hessian contributions from the dynamics module.
        """
        return self.__exact_hess_dyn

    @property
    def ext_cost_num_hess(self):
        """
        Determines if custom hessian approximation for cost contribution is used (> 0).\n
        Or if hessian contribution is evaluated exactly using CasADi external function (=0 - default).
        """
        return self.__ext_cost_num_hess

    @qp_solver.setter
    def qp_solver(self, qp_solver):
        qp_solvers = ('PARTIAL_CONDENSING_HPIPM', \
                'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', \
                'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', \
                'FULL_CONDENSING_DAQP')
        if qp_solver in qp_solvers:
            self.__qp_solver = qp_solver
        else:
            raise Exception('Invalid qp_solver value. Possible values are:\n\n' \
                    + ',\n'.join(qp_solvers) + '.\n\nYou have: ' + qp_solver + '.\n\n')

    @regularize_method.setter
    def regularize_method(self, regularize_method):
        regularize_methods = ('NO_REGULARIZE', 'MIRROR', 'PROJECT', \
                                'PROJECT_REDUC_HESS', 'CONVEXIFY')
        if regularize_method in regularize_methods:
            self.__regularize_method = regularize_method
        else:
            raise Exception('Invalid regularize_method value. Possible values are:\n\n' \
                    + ',\n'.join(regularize_methods) + '.\n\nYou have: ' + regularize_method + '.\n\n')

    @collocation_type.setter
    def collocation_type(self, collocation_type):
        collocation_types = ('GAUSS_RADAU_IIA', 'GAUSS_LEGENDRE')
        if collocation_type in collocation_types:
            self.__collocation_type = collocation_type
        else:
            raise Exception('Invalid collocation_type value. Possible values are:\n\n' \
                    + ',\n'.join(collocation_types) + '.\n\nYou have: ' + collocation_type + '.\n\n')

    @hpipm_mode.setter
    def hpipm_mode(self, hpipm_mode):
        hpipm_modes = ('BALANCE', 'SPEED_ABS', 'SPEED', 'ROBUST')
        if hpipm_mode in hpipm_modes:
            self.__hpipm_mode = hpipm_mode
        else:
            raise Exception('Invalid hpipm_mode value. Possible values are:\n\n' \
                    + ',\n'.join(hpipm_modes) + '.\n\nYou have: ' + hpipm_mode + '.\n\n')

    @ext_fun_compile_flags.setter
    def ext_fun_compile_flags(self, ext_fun_compile_flags):
        if isinstance(ext_fun_compile_flags, str):
            self.__ext_fun_compile_flags = ext_fun_compile_flags
        else:
            raise Exception('Invalid ext_fun_compile_flags, expected a string.\n')


    @custom_update_filename.setter
    def custom_update_filename(self, custom_update_filename):
        if isinstance(custom_update_filename, str):
            self.__custom_update_filename = custom_update_filename
        else:
            raise Exception('Invalid custom_update_filename, expected a string.\n')

    @custom_templates.setter
    def custom_templates(self, custom_templates):
        if not isinstance(custom_templates, list):
            raise Exception('Invalid custom_templates, expected a list.\n')
        for tup in custom_templates:
            if not isinstance(tup, tuple):
                raise Exception('Invalid custom_templates, shoubld be list of tuples.\n')
            for s in tup:
                if not isinstance(s, str):
                    raise Exception('Invalid custom_templates, shoubld be list of tuples of strings.\n')
        self.__custom_templates = custom_templates

    @custom_update_header_filename.setter
    def custom_update_header_filename(self, custom_update_header_filename):
        if isinstance(custom_update_header_filename, str):
            self.__custom_update_header_filename = custom_update_header_filename
        else:
            raise Exception('Invalid custom_update_header_filename, expected a string.\n')

    @custom_update_copy.setter
    def custom_update_copy(self, custom_update_copy):
        if isinstance(custom_update_copy, bool):
            self.__custom_update_copy = custom_update_copy
        else:
            raise Exception('Invalid custom_update_copy, expected a bool.\n')

    @hessian_approx.setter
    def hessian_approx(self, hessian_approx):
        hessian_approxs = ('GAUSS_NEWTON', 'EXACT')
        if hessian_approx in hessian_approxs:
            self.__hessian_approx = hessian_approx
        else:
            raise Exception('Invalid hessian_approx value. Possible values are:\n\n' \
                    + ',\n'.join(hessian_approxs) + '.\n\nYou have: ' + hessian_approx + '.\n\n')

    @integrator_type.setter
    def integrator_type(self, integrator_type):
        integrator_types = ('ERK', 'IRK', 'GNSF', 'DISCRETE', 'LIFTED_IRK')
        if integrator_type in integrator_types:
            self.__integrator_type = integrator_type
        else:
            raise Exception('Invalid integrator_type value. Possible values are:\n\n' \
                    + ',\n'.join(integrator_types) + '.\n\nYou have: ' + integrator_type + '.\n\n')

    @tf.setter
    def tf(self, tf):
        self.__tf = tf

    @time_steps.setter
    def time_steps(self, time_steps):
        if isinstance(time_steps, np.ndarray):
            if len(time_steps.shape) == 1:
                    self.__time_steps = time_steps
            else:
                raise Exception('Invalid time_steps, expected np.ndarray of shape (N,).')
        else:
            raise Exception('Invalid time_steps, expected np.ndarray.')

    @shooting_nodes.setter
    def shooting_nodes(self, shooting_nodes):
        if isinstance(shooting_nodes, np.ndarray):
            if len(shooting_nodes.shape) == 1:
                self.__shooting_nodes = shooting_nodes
            else:
                raise Exception('Invalid shooting_nodes, expected np.ndarray of shape (N+1,).')
        else:
            raise Exception('Invalid shooting_nodes, expected np.ndarray.')

    @Tsim.setter
    def Tsim(self, Tsim):
        self.__Tsim = Tsim

    @globalization.setter
    def globalization(self, globalization):
        globalization_types = ('MERIT_BACKTRACKING', 'FIXED_STEP')
        if globalization in globalization_types:
            self.__globalization = globalization
        else:
            raise Exception('Invalid globalization value. Possible values are:\n\n' \
                    + ',\n'.join(globalization_types) + '.\n\nYou have: ' + globalization + '.\n\n')

    @alpha_min.setter
    def alpha_min(self, alpha_min):
        self.__alpha_min = alpha_min

    @alpha_reduction.setter
    def alpha_reduction(self, alpha_reduction):
        self.__alpha_reduction = alpha_reduction

    @line_search_use_sufficient_descent.setter
    def line_search_use_sufficient_descent(self, line_search_use_sufficient_descent):
        if line_search_use_sufficient_descent in [0, 1]:
            self.__line_search_use_sufficient_descent = line_search_use_sufficient_descent
        else:
            raise Exception(f'Invalid value for line_search_use_sufficient_descent. Possible values are 0, 1, got {line_search_use_sufficient_descent}')

    @globalization_use_SOC.setter
    def globalization_use_SOC(self, globalization_use_SOC):
        if globalization_use_SOC in [0, 1]:
            self.__globalization_use_SOC = globalization_use_SOC
        else:
            raise Exception(f'Invalid value for globalization_use_SOC. Possible values are 0, 1, got {globalization_use_SOC}')

    @full_step_dual.setter
    def full_step_dual(self, full_step_dual):
        if full_step_dual in [0, 1]:
            self.__full_step_dual = full_step_dual
        else:
            raise Exception(f'Invalid value for full_step_dual. Possible values are 0, 1, got {full_step_dual}')

    @eps_sufficient_descent.setter
    def eps_sufficient_descent(self, eps_sufficient_descent):
        if isinstance(eps_sufficient_descent, float) and eps_sufficient_descent > 0:
            self.__eps_sufficient_descent = eps_sufficient_descent
        else:
            raise Exception('Invalid eps_sufficient_descent value. eps_sufficient_descent must be a positive float.')

    @sim_method_num_stages.setter
    def sim_method_num_stages(self, sim_method_num_stages):

        # if isinstance(sim_method_num_stages, int):
        #     self.__sim_method_num_stages = sim_method_num_stages
        # else:
        #     raise Exception('Invalid sim_method_num_stages value. sim_method_num_stages must be an integer.')

        self.__sim_method_num_stages = sim_method_num_stages

    @sim_method_num_steps.setter
    def sim_method_num_steps(self, sim_method_num_steps):

        # if isinstance(sim_method_num_steps, int):
        #     self.__sim_method_num_steps = sim_method_num_steps
        # else:
        #     raise Exception('Invalid sim_method_num_steps value. sim_method_num_steps must be an integer.')
        self.__sim_method_num_steps = sim_method_num_steps


    @sim_method_newton_iter.setter
    def sim_method_newton_iter(self, sim_method_newton_iter):

        if isinstance(sim_method_newton_iter, int):
            self.__sim_method_newton_iter = sim_method_newton_iter
        else:
            raise Exception('Invalid sim_method_newton_iter value. sim_method_newton_iter must be an integer.')

    @sim_method_jac_reuse.setter
    def sim_method_jac_reuse(self, sim_method_jac_reuse):
        # if sim_method_jac_reuse in (True, False):
        self.__sim_method_jac_reuse = sim_method_jac_reuse
        # else:
            # raise Exception('Invalid sim_method_jac_reuse value. sim_method_jac_reuse must be a Boolean.')

    @nlp_solver_type.setter
    def nlp_solver_type(self, nlp_solver_type):
        nlp_solver_types = ('SQP', 'SQP_RTI')
        if nlp_solver_type in nlp_solver_types:
            self.__nlp_solver_type = nlp_solver_type
        else:
            raise Exception('Invalid nlp_solver_type value. Possible values are:\n\n' \
                    + ',\n'.join(nlp_solver_types) + '.\n\nYou have: ' + nlp_solver_type + '.\n\n')

    @nlp_solver_step_length.setter
    def nlp_solver_step_length(self, nlp_solver_step_length):
        if isinstance(nlp_solver_step_length, float) and nlp_solver_step_length > 0:
            self.__nlp_solver_step_length = nlp_solver_step_length
        else:
            raise Exception('Invalid nlp_solver_step_length value. nlp_solver_step_length must be a positive float.')

    @levenberg_marquardt.setter
    def levenberg_marquardt(self, levenberg_marquardt):
        if isinstance(levenberg_marquardt, float) and levenberg_marquardt >= 0:
            self.__levenberg_marquardt = levenberg_marquardt
        else:
            raise Exception('Invalid levenberg_marquardt value. levenberg_marquardt must be a positive float.')

    @qp_solver_iter_max.setter
    def qp_solver_iter_max(self, qp_solver_iter_max):
        if isinstance(qp_solver_iter_max, int) and qp_solver_iter_max > 0:
            self.__qp_solver_iter_max = qp_solver_iter_max
        else:
            raise Exception('Invalid qp_solver_iter_max value. qp_solver_iter_max must be a positive int.')

    @qp_solver_ric_alg.setter
    def qp_solver_ric_alg(self, qp_solver_ric_alg):
        if qp_solver_ric_alg in [0, 1]:
            self.__qp_solver_ric_alg = qp_solver_ric_alg
        else:
            raise Exception(f'Invalid qp_solver_ric_alg value. qp_solver_ric_alg must be in [0, 1], got {qp_solver_ric_alg}.')

    @qp_solver_cond_ric_alg.setter
    def qp_solver_cond_ric_alg(self, qp_solver_cond_ric_alg):
        if qp_solver_cond_ric_alg in [0, 1]:
            self.__qp_solver_cond_ric_alg = qp_solver_cond_ric_alg
        else:
            raise Exception(f'Invalid qp_solver_cond_ric_alg value. qp_solver_cond_ric_alg must be in [0, 1], got {qp_solver_cond_ric_alg}.')


    @qp_solver_cond_N.setter
    def qp_solver_cond_N(self, qp_solver_cond_N):
        if isinstance(qp_solver_cond_N, int) and qp_solver_cond_N >= 0:
            self.__qp_solver_cond_N = qp_solver_cond_N
        else:
            raise Exception('Invalid qp_solver_cond_N value. qp_solver_cond_N must be a positive int.')

    @qp_solver_warm_start.setter
    def qp_solver_warm_start(self, qp_solver_warm_start):
        if qp_solver_warm_start in [0, 1, 2]:
            self.__qp_solver_warm_start = qp_solver_warm_start
        else:
            raise Exception('Invalid qp_solver_warm_start value. qp_solver_warm_start must be 0 or 1 or 2.')

    @qp_tol.setter
    def qp_tol(self, qp_tol):
        if isinstance(qp_tol, float) and qp_tol > 0:
            self.__qp_solver_tol_eq = qp_tol
            self.__qp_solver_tol_ineq = qp_tol
            self.__qp_solver_tol_stat = qp_tol
            self.__qp_solver_tol_comp = qp_tol
        else:
            raise Exception('Invalid qp_tol value. qp_tol must be a positive float.')

    @qp_solver_tol_stat.setter
    def qp_solver_tol_stat(self, qp_solver_tol_stat):
        if isinstance(qp_solver_tol_stat, float) and qp_solver_tol_stat > 0:
            self.__qp_solver_tol_stat = qp_solver_tol_stat
        else:
            raise Exception('Invalid qp_solver_tol_stat value. qp_solver_tol_stat must be a positive float.')

    @qp_solver_tol_eq.setter
    def qp_solver_tol_eq(self, qp_solver_tol_eq):
        if isinstance(qp_solver_tol_eq, float) and qp_solver_tol_eq > 0:
            self.__qp_solver_tol_eq = qp_solver_tol_eq
        else:
            raise Exception('Invalid qp_solver_tol_eq value. qp_solver_tol_eq must be a positive float.')

    @qp_solver_tol_ineq.setter
    def qp_solver_tol_ineq(self, qp_solver_tol_ineq):
        if isinstance(qp_solver_tol_ineq, float) and qp_solver_tol_ineq > 0:
            self.__qp_solver_tol_ineq = qp_solver_tol_ineq
        else:
            raise Exception('Invalid qp_solver_tol_ineq value. qp_solver_tol_ineq must be a positive float.')

    @qp_solver_tol_comp.setter
    def qp_solver_tol_comp(self, qp_solver_tol_comp):
        if isinstance(qp_solver_tol_comp, float) and qp_solver_tol_comp > 0:
            self.__qp_solver_tol_comp = qp_solver_tol_comp
        else:
            raise Exception('Invalid qp_solver_tol_comp value. qp_solver_tol_comp must be a positive float.')

    @tol.setter
    def tol(self, tol):
        if isinstance(tol, float) and tol > 0:
            self.__nlp_solver_tol_eq = tol
            self.__nlp_solver_tol_ineq = tol
            self.__nlp_solver_tol_stat = tol
            self.__nlp_solver_tol_comp = tol
        else:
            raise Exception('Invalid tol value. tol must be a positive float.')

    @nlp_solver_tol_stat.setter
    def nlp_solver_tol_stat(self, nlp_solver_tol_stat):
        if isinstance(nlp_solver_tol_stat, float) and nlp_solver_tol_stat > 0:
            self.__nlp_solver_tol_stat = nlp_solver_tol_stat
        else:
            raise Exception('Invalid nlp_solver_tol_stat value. nlp_solver_tol_stat must be a positive float.')

    @nlp_solver_tol_eq.setter
    def nlp_solver_tol_eq(self, nlp_solver_tol_eq):
        if isinstance(nlp_solver_tol_eq, float) and nlp_solver_tol_eq > 0:
            self.__nlp_solver_tol_eq = nlp_solver_tol_eq
        else:
            raise Exception('Invalid nlp_solver_tol_eq value. nlp_solver_tol_eq must be a positive float.')

    @nlp_solver_tol_ineq.setter
    def nlp_solver_tol_ineq(self, nlp_solver_tol_ineq):
        if isinstance(nlp_solver_tol_ineq, float) and nlp_solver_tol_ineq > 0:
            self.__nlp_solver_tol_ineq = nlp_solver_tol_ineq
        else:
            raise Exception('Invalid nlp_solver_tol_ineq value. nlp_solver_tol_ineq must be a positive float.')

    @nlp_solver_ext_qp_res.setter
    def nlp_solver_ext_qp_res(self, nlp_solver_ext_qp_res):
        if nlp_solver_ext_qp_res in [0, 1]:
            self.__nlp_solver_ext_qp_res = nlp_solver_ext_qp_res
        else:
            raise Exception('Invalid nlp_solver_ext_qp_res value. nlp_solver_ext_qp_res must be in [0, 1].')

    @nlp_solver_tol_comp.setter
    def nlp_solver_tol_comp(self, nlp_solver_tol_comp):
        if isinstance(nlp_solver_tol_comp, float) and nlp_solver_tol_comp > 0:
            self.__nlp_solver_tol_comp = nlp_solver_tol_comp
        else:
            raise Exception('Invalid nlp_solver_tol_comp value. nlp_solver_tol_comp must be a positive float.')

    @nlp_solver_max_iter.setter
    def nlp_solver_max_iter(self, nlp_solver_max_iter):

        if isinstance(nlp_solver_max_iter, int) and nlp_solver_max_iter > 0:
            self.__nlp_solver_max_iter = nlp_solver_max_iter
        else:
            raise Exception('Invalid nlp_solver_max_iter value. nlp_solver_max_iter must be a positive int.')

    @print_level.setter
    def print_level(self, print_level):
        if isinstance(print_level, int) and print_level >= 0:
            self.__print_level = print_level
        else:
            raise Exception('Invalid print_level value. print_level takes one of the values >=0.')

    @model_external_shared_lib_dir.setter
    def model_external_shared_lib_dir(self, model_external_shared_lib_dir):
        if isinstance(model_external_shared_lib_dir, str) :
            self.__model_external_shared_lib_dir = model_external_shared_lib_dir
        else:
            raise Exception('Invalid model_external_shared_lib_dir value. Str expected.' \
            + '.\n\nYou have: ' + type(model_external_shared_lib_dir) + '.\n\n')

    @model_external_shared_lib_name.setter
    def model_external_shared_lib_name(self, model_external_shared_lib_name):
        if isinstance(model_external_shared_lib_name, str) :
            if model_external_shared_lib_name[-3:] == '.so' : 
                raise Exception('Invalid model_external_shared_lib_name value. Remove the .so extension.' \
            + '.\n\nYou have: ' + type(model_external_shared_lib_name) + '.\n\n')
            else :
                self.__model_external_shared_lib_name = model_external_shared_lib_name
        else:
            raise Exception('Invalid model_external_shared_lib_name value. Str expected.' \
            + '.\n\nYou have: ' + type(model_external_shared_lib_name) + '.\n\n')

    @exact_hess_constr.setter
    def exact_hess_constr(self, exact_hess_constr):
        if exact_hess_constr in [0, 1]:
            self.__exact_hess_constr = exact_hess_constr
        else:
            raise Exception('Invalid exact_hess_constr value. exact_hess_constr takes one of the values 0, 1.')

    @exact_hess_cost.setter
    def exact_hess_cost(self, exact_hess_cost):
        if exact_hess_cost in [0, 1]:
            self.__exact_hess_cost = exact_hess_cost
        else:
            raise Exception('Invalid exact_hess_cost value. exact_hess_cost takes one of the values 0, 1.')

    @exact_hess_dyn.setter
    def exact_hess_dyn(self, exact_hess_dyn):
        if exact_hess_dyn in [0, 1]:
            self.__exact_hess_dyn = exact_hess_dyn
        else:
            raise Exception('Invalid exact_hess_dyn value. exact_hess_dyn takes one of the values 0, 1.')

    @ext_cost_num_hess.setter
    def ext_cost_num_hess(self, ext_cost_num_hess):
        if ext_cost_num_hess in [0, 1]:
            self.__ext_cost_num_hess = ext_cost_num_hess
        else:
            raise Exception('Invalid ext_cost_num_hess value. ext_cost_num_hess takes one of the values 0, 1.')

    def set(self, attr, value):
        setattr(self, attr, value)


class AcadosOcp:
    """
    Class containing the full description of the optimal control problem.
    This object can be used to create an :py:class:`acados_template.acados_ocp_solver.AcadosOcpSolver`.

    The class has the following properties that can be modified to formulate a specific OCP, see below:

        - :py:attr:`dims` of type :py:class:`acados_template.acados_ocp.AcadosOcpDims`
        - :py:attr:`model` of type :py:class:`acados_template.acados_model.AcadosModel`
        - :py:attr:`cost` of type :py:class:`acados_template.acados_ocp.AcadosOcpCost`
        - :py:attr:`constraints` of type :py:class:`acados_template.acados_ocp.AcadosOcpConstraints`
        - :py:attr:`solver_options` of type :py:class:`acados_template.acados_ocp.AcadosOcpOptions`

        - :py:attr:`acados_include_path` (set automatically)
        - :py:attr:`shared_lib_ext` (set automatically)
        - :py:attr:`acados_lib_path` (set automatically)
        - :py:attr:`parameter_values` - used to initialize the parameters (can be changed)
    """
    def __init__(self, acados_path=''):
        """
        Keyword arguments:
        acados_path -- path of your acados installation
        """
        if acados_path == '':
            acados_path = get_acados_path()

        self.dims = AcadosOcpDims()
        """Dimension definitions, type :py:class:`acados_template.acados_ocp.AcadosOcpDims`"""
        self.model = AcadosModel()
        """Model definitions, type :py:class:`acados_template.acados_model.AcadosModel`"""
        self.cost = AcadosOcpCost()
        """Cost definitions, type :py:class:`acados_template.acados_ocp.AcadosOcpCost`"""
        self.constraints = AcadosOcpConstraints()
        """Constraints definitions, type :py:class:`acados_template.acados_ocp.AcadosOcpConstraints`"""
        self.solver_options = AcadosOcpOptions()
        """Solver Options, type :py:class:`acados_template.acados_ocp.AcadosOcpOptions`"""

        self.acados_include_path = os.path.join(acados_path, 'include').replace(os.sep, '/') # the replace part is important on Windows for CMake
        """Path to acados include directory (set automatically), type: `string`"""
        self.acados_lib_path = os.path.join(acados_path, 'lib').replace(os.sep, '/') # the replace part is important on Windows for CMake
        """Path to where acados library is located, type: `string`"""
        self.shared_lib_ext = get_lib_ext()

        # get cython paths
        from sysconfig import get_paths
        self.cython_include_dirs = [np.get_include(), get_paths()['include']]

        self.__parameter_values = np.array([])
        self.__problem_class = 'OCP'

        self.code_export_directory = 'c_generated_code'
        """Path to where code will be exported. Default: `c_generated_code`."""

    @property
    def parameter_values(self):
        """:math:`p` - initial values for parameter - can be updated stagewise"""
        return self.__parameter_values

    @parameter_values.setter
    def parameter_values(self, parameter_values):
        if isinstance(parameter_values, np.ndarray):
            self.__parameter_values = parameter_values
        else:
            raise Exception('Invalid parameter_values value. ' +
                            f'Expected numpy array, got {type(parameter_values)}.')

    def set(self, attr, value):
        # tokenize string
        tokens = attr.split('_', 1)
        if len(tokens) > 1:
            setter_to_call = getattr(getattr(self, tokens[0]), 'set')
        else:
            setter_to_call = getattr(self, 'set')

        setter_to_call(tokens[1], value)

        return
