import rclpy
import math
import threading
import socket
import pickle
import sys
sys.path.append("/home/kevinm/Documents/projects/drone/openpilot")
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from openpilot.tools.sim.bridge.gz.msgs import Image as gzImage, Odometry, GPSPos, Status, RequestType, Report
from openpilot.tools.sim.lib.common import W, H
from px4_msgs.msg import VehicleOdometry, SensorGps, VehicleStatus, VehicleLocalPosition
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
import numpy as np
import cv2

class Server(Node):
  def __init__(self):
    super().__init__("rosserver")
    self.spawned_threads = dict()
    self.next_thread_id = 0
    self.image_sub_ = self.create_subscription(Image, "/camera", self.img_cb, 10)

    qos = QoSProfile(
      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
      reliability=QoSReliabilityPolicy.BEST_EFFORT,
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=1)
    qos_profile_pub = QoSProfile(
      durability=QoSDurabilityPolicy.VOLATILE,
      reliability=QoSReliabilityPolicy.BEST_EFFORT,
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=10)
    self.middle_tire_r = None
    self.odometry_sub_ = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.odometry_cb, qos)
    self.gps_sub_ = self.create_subscription(SensorGps, "/fmu/out/vehicle_gps_position", self.gps_cb, qos)
    self.status_sub_ = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.status_cb, qos)
    self.local_pos_sub_ = self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_pos_cb, qos)
    self.offboard_mode_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile_pub)
    self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_profile_pub)
    self.vehicle_command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_profile_pub)

    self.image = gzImage()
    self.odometry = Odometry([0,0,0], 0)
    self.gps = GPSPos(0,0,0)
    self.status = Status(False, False)
    self.delta_heading = 0

    self.main_thread = threading.Thread(target=self.run)
    self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server.bind(("0.0.0.0", 8069))
    self.main_thread.start()

    self.interval = 0.1
    self.tires_steering_angle = 0.0
    self.create_timer(self.interval, self.publish)
    self.step = 0
    self.pose = [0.0, 0.0, 0.0]
    self.velocity = [0.0, 0.0, 0.0]
    self.orientation = [0.0, 0.0, 0.0, 1.0]
    self.desired_yaw = 0
    self.accel = 0
    self.desired_z = None
    self.get_logger().info("rosserver started")

  def __del__(self):
    self.server.close()
    for thread in self.spawned_threads.values():
      thread.join()
    self.main_thread.join()

  def publish_vehicle_command(self, command, param1, param2):
      msg = VehicleCommand()
      msg.command = command
      msg.param1 = param1
      msg.param2 = param2
      msg.target_system = 1
      msg.target_component = 1
      msg.source_system = 1
      msg.source_component = 0
      msg.from_external = True
      msg.timestamp = (self.get_clock().now().nanoseconds // 1000)
      self.vehicle_command_pub.publish(msg)

  def publish_trajectory_setpoint(self, accel, middle_tire_r):
      msg = TrajectorySetpoint()
      msg.position[0] = self.pose[0]
      msg.position[1] = self.pose[1]
      msg.position[2] = -3.5 if self.desired_z is None else self.desired_z

      if self.middle_tire_r is None:
        self.trajectory_setpoint_pub.publish(msg)
        return

      linear_distance =  (self.velocity[0] + accel * self.interval) * self.interval
      circunference = 2 * math.pi * middle_tire_r
      if middle_tire_r == 0:
        yaw = 0
      else:
        yaw =(-1 if self.tires_steering_angle >= 0 else 1) * math.radians(linear_distance * 360 / circunference)

      if yaw > math.pi/4:
        yaw = math.pi/4
      elif yaw < -math.pi/4:
        yaw = -math.pi/4

      rotation_matrix = [[math.cos(yaw), -math.sin(yaw), 0],
                         [math.sin(yaw), math.cos(yaw), 0],
                         [0, 0, 1]]


      msg.position = [rotation_matrix[0][0] * msg.position[0] + rotation_matrix[0][1] * msg.position[1] + rotation_matrix[0][2] * msg.position[2],
                      rotation_matrix[1][0] * msg.position[0] + rotation_matrix[1][1] * msg.position[1] + rotation_matrix[1][2] * msg.position[2],
                      rotation_matrix[2][0] * msg.position[0] + rotation_matrix[2][1] * msg.position[1] + rotation_matrix[2][2] * msg.position[2]]

      self.desired_yaw = yaw
      msg.yaw = self.get_current_yaw() + self.desired_yaw
      rotation_matrix = [[math.cos(msg.yaw), -math.sin(msg.yaw), 0],
                         [math.sin(msg.yaw), math.cos(msg.yaw), 0],
                         [0, 0, 1]]

      throttle = [linear_distance, 0, 0]
      throttle = [rotation_matrix[0][0] * throttle[0] + rotation_matrix[0][1] * throttle[1] + rotation_matrix[0][2] * throttle[2],
                  rotation_matrix[1][0] * throttle[0] + rotation_matrix[1][1] * throttle[1] + rotation_matrix[1][2] * throttle[2],
                  rotation_matrix[2][0] * throttle[0] + rotation_matrix[2][1] * throttle[1] + rotation_matrix[2][2] * throttle[2]]

      msg.position[0] += throttle[0]
      msg.position[1] += throttle[1]
      msg.position[2] += throttle[2]

#      currYaw = math.atan2((msg.position[1] - self.pose[1]),
#                          (msg.position[0] - self.pose[0]));

#      delta = [linear_distance, 0, 0]
#      rotation_matrix = [[math.cos(msg.yaw), -math.sin(msg.yaw), 0],
#                         [math.sin(msg.yaw), math.cos(msg.yaw), 0],
#                         [0, 0, 1]]
#      rotated_delta = [rotation_matrix[0][0] * delta[0] + rotation_matrix[0][1] * delta[1] + rotation_matrix[0][2] * delta[2],
#                       rotation_matrix[1][0] * delta[0] + rotation_matrix[1][1] * delta[1] + rotation_matrix[1][2] * delta[2],
#                       rotation_matrix[2][0] * delta[0] + rotation_matrix[2][1] * delta[1] + rotation_matrix[2][2] * delta[2]]
#      msg.position[0] += rotated_delta[0]
#      msg.position[1] += rotated_delta[1]
#      msg.position[2] += rotated_delta[2]
      msg.timestamp = (self.get_clock().now().nanoseconds // 1000)
      self.trajectory_setpoint_pub.publish(msg)

  def publish_offboard_control_mode(self):
      msg = OffboardControlMode()
      msg.position = True
      msg.velocity = False
      msg.acceleration = False
      msg.attitude = False
      msg.body_rate = False
      msg.timestamp = (self.get_clock().now().nanoseconds // 1000)
      self.offboard_mode_pub.publish(msg)

  def get_current_yaw(self):
      w = self.orientation[0]
      x = self.orientation[1]
      y = self.orientation[2]
      z = self.orientation[3]
      return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

  def publish(self):
    self.publish_offboard_control_mode()
    if (self.step == 10):
        self.get_logger().info("Changing to offboard mode")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.arm()

    self.publish_trajectory_setpoint(self.accel, self.middle_tire_r)
    self.step += 1



  def run(self):
    self.server.listen()
    while rclpy.ok():
      conn, addr = self.server.accept()
      client_thread = threading.Thread(target=self.handle_client, args=(conn, addr, int(self.next_thread_id)))
      self.spawned_threads[self.next_thread_id]=client_thread
      self.next_thread_id += 1
      client_thread.start()

  def handle_client(self, conn, addr, thread_id):
    length = conn.recv(4)
    if(len(length) == 0):
      conn.close()
      self.spawned_threads.pop(thread_id)
      return

    length = int.from_bytes(length, "little")
    data = conn.recv(length)
    msg = pickle.loads(data)
    response = None
    if msg.type == RequestType.GET_IMAGE:
      response = pickle.dumps(self.image)
    elif msg.type == RequestType.GET_ODOMETRY:
      response = pickle.dumps(self.odometry)
    elif msg.type == RequestType.GET_GPS:
      response = pickle.dumps(self.gps)
    elif msg.type == RequestType.GET_STATUS:
      response = pickle.dumps(self.status)
    elif msg.type == RequestType.GET_REPORT:
      response = pickle.dumps(Report(self.image, self.gps, self.odometry, self.status))
    elif msg.type == RequestType.IN_COMMAND:
      b=10
      c=4
      a=6
      self.tires_steering_angle = msg.data[0]/17
      angle = abs(math.radians(np.clip(self.tires_steering_angle,-44,44)))
      if angle == 0:
        self.middle_tire_r = 0
      else:
        int_tire_r = b/math.sin(angle) - ((a-c)/2)
        out_tire_r = b/math.tan(angle) - ((a-c)/2)
        self.middle_tire_r = (int_tire_r + out_tire_r)/2
      self.accel = msg.data[1] * 150
      response = b'OK'

    conn.send(len(response).to_bytes(4, "little"))
    conn.sendall(response)

    conn.close()
    self.spawned_threads.pop(thread_id)


  def img_cb(self, msg: Image):
    self.image = gzImage(msg.data, msg.height, msg.width)

  def local_pos_cb(self, msg: VehicleLocalPosition):
    self.delta_heading = msg.delta_heading
#    self.odometry.heading = self.desired_yaw

  def odometry_cb(self, msg: VehicleOdometry):
    if self.desired_z is None:
      self.desired_z = msg.position[2] - 2
    self.pose = msg.position
    self.orientation = msg.q
    self.velocity = msg.velocity
    self.odometry = Odometry(msg.velocity, self.get_current_yaw(), self.desired_yaw)

  def gps_cb(self, msg: SensorGps):
    self.gps = GPSPos(msg.latitude_deg, msg.longitude_deg, msg.altitude_msl_m)

  def status_cb(self, msg: VehicleStatus):
    self.status = Status(msg.armed_time > 0, msg.takeoff_time > 0)

  def arm(self):
      self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
      self.get_logger().info("Arm command sent")


def main(args=None):
  rclpy.init(args=args)
  node = Server()
  rclpy.spin(node)
  node.destroy_node()
  rclpy.shutdown()

if __name__ == "__main__":
  main()
