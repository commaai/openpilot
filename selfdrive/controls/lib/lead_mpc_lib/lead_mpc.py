#!/usr/bin/env python3
import numpy as np

from selfdrive.swaglog import cloudlog
from common.numpy_fast import interp
from common.realtime import sec_since_boot
from selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import generate_code, get_default_simulink_options
from selfdrive.modeld.constants import T_IDXS
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
import os
import sys
from common.basedir import BASEDIR

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sqrt, exp
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from selfdrive.controls.lib.drive_helpers import CONTROL_N

# TODO: clean this up
acados_path = os.path.join(BASEDIR, "phonelibs/acados/x86_64")
os.environ["TERA_PATH"] = os.path.join(acados_path, "t_renderer")
sys.path.append(os.path.join(BASEDIR, "pyextra"))



LEAD_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = "acados_ocp_lead.json"

MPC_T = list(np.arange(0,1.,.2)) + list(np.arange(1.,10.6,.6))
N = len(MPC_T) - 1

def gen_lead_model():
  model = AcadosModel()
  model.name = 'lead'

  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # live parameters
  x_lead = SX.sym('x_lead')
  v_lead = SX.sym('v_lead')
  model.p = vertcat(x_lead, v_lead)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_lead_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_lead_model()

  Tf = np.array(MPC_T)[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.diag([0.0, 0.0, 0.0, 0.0])
  Q = np.diag([0.0, 0.0, 0.0])

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  ocp.cost.yref = np.zeros((4, ))
  ocp.cost.yref_e = np.zeros((3, ))

  x_lead, v_lead = ocp.model.p[0], ocp.model.p[1]
  G = 9.81
  TR = 1.8
  desired_dist = (v_ego * TR
                  - (v_lead - v_ego) * TR
                  + v_ego*v_ego/(2*G)
                  - v_lead * v_lead / (2*G))
  dist_err = (desired_dist + 4.0 - (x_lead - x_ego))/(sqrt(v_ego + 0.5) + 0.1)

  # TODO hacky weights to keep behavior the same
  ocp.model.cost_y_expr = vertcat(exp(.3 * dist_err) - 1.,
                                  ((x_lead - x_ego) - (desired_dist + 4.0)) / (0.05 * v_ego + 0.5),
                                  a_ego * (.1 * v_ego + 1.0),
                                  j_ego * (.1 * v_ego + 1.0))
  ocp.model.cost_y_expr_e = vertcat(exp(.3 * dist_err) - 1.,
                                  ((x_lead - x_ego) - (desired_dist + 4.0)) / (0.05 * v_ego + 0.5),
                                  a_ego * (.1 * v_ego + 1.0))
  ocp.parameter_values = np.array([0., .0])

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  ocp.constraints.idxbx = np.array([1,])
  ocp.constraints.lbx = np.array([0,])
  ocp.constraints.ubx = np.array([100.,])
  x0 = np.array([0.0, 0.0, 0.0])
  ocp.constraints.x0 = x0


  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'
  ocp.solver_options.nlp_solver_tol_stat = 1e-3
  ocp.solver_options.tol = 1e-3

  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3
  #ocp.solver_options.qp_solver_warm_start = 2

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = np.array(MPC_T)

  ocp.code_export_directory = os.path.join(LEAD_MPC_DIR, "c_generated_code")
  return ocp


class LeadMpc():
  def __init__(self, lead_id):
    self.lead_id = lead_id
    ocp = gen_lead_mpc_solver()
    self.solver = AcadosOcpSolver(ocp, json_file=JSON_FILE, build=False,
                                  simulink_opts=get_default_simulink_options())
    self.x_sol = np.zeros((N+1, 3))
    self.u_sol = np.zeros((N))
    self.set_weights()

    self.v_solution = [0.0 for i in range(N)]
    self.a_solution = [0.0 for i in range(N)]
    self.j_solution = [0.0 for i in range(N-1)]
    self.last_cloudlog_t = 0
    self.status = False
    self.new_lead = False
    self.prev_lead_status = False
    self.prev_lead_x = 10

  def set_weights(self):
    W = np.diag([MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                 MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK])
    Ws = np.tile(W[None], reps=(N,1,1))
    self.solver.cost_set_slice(0, N, 'W', Ws, api='old')
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3./5.)*W[:3,:3])

  def set_cur_state(self, v, a):
    self.x0 = np.array([0, v, a])

  def update(self, carstate, radarstate, v_cruise):
    v_ego = carstate.vEgo
    if self.lead_id == 0:
      lead = radarstate.leadOne
    else:
      lead = radarstate.leadTwo
    self.status = lead.status

    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK
      print(a_lead)

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.a_lead_tau = lead.aLeadTau
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.new_lead = True

      self.prev_lead_status = True
      self.prev_lead_x = x_lead
    else:
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    dt =.2
    t = .0
    x_ego = 0
    v_ego_e = v_ego
    a_lead_0 = a_lead
    x_lead_0 = x_lead
    v_lead_0 = v_lead
    ps = np.zeros((N+1,2))
    for i in range(N+1):
      if i > 4:
        dt = .6
      ps[i] = np.array([x_lead, v_lead])
      self.solver.set(i, "p", ps[i])
      #if self.x_sol[i,0] > x_lead:
      #  new_x = np.array([x_lead - 4.0, v_lead, 0.0])
      #  self.solver.set(i, "x", new_x)
      x_ego += v_ego_e * dt
      v_ego_e = max(v_ego_e-3.0 * dt, 0.0)
      a_lead = a_lead_0 * np.exp(-self.a_lead_tau * (t**2)/2.)
      x_lead += v_lead * dt
      v_lead += a_lead * dt
      if v_lead < 0.0:
        a_lead = 0.0
        v_lead = 0.0
      t += dt

    yref = np.zeros((N+1,4))
    self.solver.cost_set_slice(0, N, "yref", yref[:N])
    self.solver.set(N, "yref", yref[N][:3])

    self.solver.solve()
    self.x_sol = self.solver.get_slice(0, N+1, 'x')
    self.u_sol = self.solver.get_slice(0, N, 'u')
    self.cost = self.solver.get_cost()
    #self.solver.print_statistics()

    self.v_solution = interp(T_IDXS[:CONTROL_N], MPC_T, list(self.x_sol[:,1]))
    self.a_solution = interp(T_IDXS[:CONTROL_N], MPC_T, list(self.x_sol[:,2]))
    self.j_solution = interp(T_IDXS[:CONTROL_N], MPC_T[:-1], list(self.u_sol[:,0]))

    # Reset if NaN or goes through lead car
    nans = np.any(np.isnan(self.x_sol))
    crashing = np.any(ps[:,0] - self.x_sol[:,0] < 0)

    print('HEYYY DOG')
    print(ps[:,0] - self.x_sol[:,0])
    print(v_ego, x_lead_0, v_lead_0, a_lead_0)
    print(self.a_solution[6])
    print(ps)
    print(self.x_sol)
    print()
    print()
    print()

    t = sec_since_boot()
    if ((crashing) and self.prev_lead_status) or nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - crashing: %s nan: %s" % (
                          self.lead_id, crashing, nans))

      self.prev_lead_status = False


if __name__ == "__main__":
  ocp = gen_lead_mpc_solver()
  generate_code(ocp, json_file=JSON_FILE)
