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


class AcadosModel():
    """
    Class containing all the information to code generate the external CasADi functions
    that are needed when creating an acados ocp solver or acados integrator.
    Thus, this class contains:

    a) the :py:attr:`name` of the model,
    b) all CasADi variables/expressions needed in the CasADi function generation process.
    """
    def __init__(self):
        ## common for OCP and Integrator
        self.name = None
        """
        The model name is used for code generation. Type: string. Default: :code:`None`
        """
        self.x = None           #: CasADi variable describing the state of the system; Default: :code:`None`
        self.xdot = None        #: CasADi variable describing the derivative of the state wrt time; Default: :code:`None`
        self.u = None           #: CasADi variable describing the input of the system; Default: :code:`None`
        self.z = []             #: CasADi variable describing the algebraic variables of the DAE; Default: :code:`empty`
        self.p = []             #: CasADi variable describing parameters of the DAE; Default: :code:`empty`
        # dynamics
        self.f_impl_expr = None
        """
        CasADi expression for the implicit dynamics :math:`f_\\text{impl}(\dot{x}, x, u, z, p) = 0`.
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.integrator_type` == 'IRK'.
        Default: :code:`None`
        """
        self.f_expl_expr = None
        """
        CasADi expression for the explicit dynamics :math:`\dot{x} = f_\\text{expl}(x, u, p)`.
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.integrator_type` == 'ERK'.
        Default: :code:`None`
        """
        self.disc_dyn_expr = None
        """
        CasADi expression for the discrete dynamics :math:`x_{+} = f_\\text{disc}(x, u, p)`.
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.integrator_type` == 'DISCRETE'.
        Default: :code:`None`
        """

        self.dyn_ext_fun_type = 'casadi'  #: type of external functions for dynamics module; 'casadi' or 'generic'; Default: 'casadi'
        self.dyn_generic_source = None  #: name of source file for discrete dyanamics; Default: :code:`None`
        self.dyn_disc_fun_jac_hess = None  #: name of function discrete dyanamics + jacobian and hessian; Default: :code:`None`
        self.dyn_disc_fun_jac = None  #: name of function discrete dyanamics + jacobian; Default: :code:`None`
        self.dyn_disc_fun = None  #: name of function discrete dyanamics; Default: :code:`None`

        # for GNSF models
        self.gnsf = {'nontrivial_f_LO': 1, 'purely_linear': 0}
        """
        dictionary containing information on GNSF structure needed when rendering templates.
        Contains integers `nontrivial_f_LO`, `purely_linear`.
        """

        ## for OCP
        # constraints
        # BGH(default): lh <= h(x, u) <= uh
        self.con_h_expr = None  #: CasADi expression for the constraint :math:`h`; Default: :code:`None`
        # BGP(convex over nonlinear): lphi <= phi(r(x, u)) <= uphi
        self.con_phi_expr = None  #: CasADi expression for the constraint phi; Default: :code:`None`
        self.con_r_expr = None  #: CasADi expression for the constraint phi(r); Default: :code:`None`
        self.con_r_in_phi = None
        # terminal
        self.con_h_expr_e = None  #: CasADi expression for the terminal constraint :math:`h^e`; Default: :code:`None`
        self.con_r_expr_e = None  #: CasADi expression for the terminal constraint; Default: :code:`None`
        self.con_phi_expr_e = None  #: CasADi expression for the terminal constraint; Default: :code:`None`
        self.con_r_in_phi_e = None
        # cost
        self.cost_y_expr = None  #: CasADi expression for nonlinear least squares; Default: :code:`None`
        self.cost_y_expr_e = None  #: CasADi expression for nonlinear least squares, terminal; Default: :code:`None`
        self.cost_y_expr_0 = None  #: CasADi expression for nonlinear least squares, initial; Default: :code:`None`
        self.cost_expr_ext_cost = None  #: CasADi expression for external cost; Default: :code:`None`
        self.cost_expr_ext_cost_e = None  #: CasADi expression for external cost, terminal; Default: :code:`None`
        self.cost_expr_ext_cost_0 = None  #: CasADi expression for external cost, initial; Default: :code:`None`
        self.cost_expr_ext_cost_custom_hess = None  #: CasADi expression for custom hessian (only for external cost); Default: :code:`None`
        self.cost_expr_ext_cost_custom_hess_e = None  #: CasADi expression for custom hessian (only for external cost), terminal; Default: :code:`None`
        self.cost_expr_ext_cost_custom_hess_0 = None  #: CasADi expression for custom hessian (only for external cost), initial; Default: :code:`None`

        ## CONVEX_OVER_NONLINEAR convex-over-nonlinear cost: psi(y(x, u, p) - y_ref; p)
        self.cost_psi_expr_0 = None
        """
        CasADi expression for the outer loss function :math:`\psi(r, p)`, initial; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type_0` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_psi_expr = None
        """
        CasADi expression for the outer loss function :math:`\psi(r, p)`; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_psi_expr_e = None
        """
        CasADi expression for the outer loss function :math:`\psi(r, p)`, terminal; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type_e` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_r_in_psi_expr_0 = None
        """
        CasADi expression for the argument :math:`r`; to the outer loss function :math:`\psi(r, p)`, initial; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type_0` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_r_in_psi_expr = None
        """
        CasADi expression for the argument :math:`r`; to the outer loss function :math:`\psi(r, p)`; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_r_in_psi_expr_e = None
        """
        CasADi expression for the argument :math:`r`; to the outer loss function :math:`\psi(r, p)`, terminal; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type_e` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_conl_custom_outer_hess_0 = None
        """
        CasADi expression for the custom hessian of the outer loss function (only for convex-over-nonlinear cost), initial; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type_0` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_conl_custom_outer_hess = None
        """
        CasADi expression for the custom hessian of the outer loss function (only for convex-over-nonlinear cost); Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type` == 'CONVEX_OVER_NONLINEAR'.
        """
        self.cost_conl_custom_outer_hess_e = None
        """
        CasADi expression for the custom hessian of the outer loss function (only for convex-over-nonlinear cost), terminal; Default: :code:`None`
        Used if :py:attr:`acados_template.acados_ocp.AcadosOcpOptions.cost_type_e` == 'CONVEX_OVER_NONLINEAR'.
        """
