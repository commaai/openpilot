
Reference Frames
------
Many reference frames are used throughout. This
folder contains all helper functions needed to
transform between them. Generally this is done
by generating a rotation matrix and multiplying.


| Name | [x, y, z] | Units | Notes |
| :-------------: |:-------------:| :-----:| :----: |
| Geodetic | [Latitude, Longitude, Altitude] | geodetic coordinates | Sometimes used as [lon, lat, alt], avoid this frame. |
| ECEF | [x, y, z] | meters | We use **ITRF14 (IGS14)**, NOT NAD83. <br> This is the global Mesh3D frame. |
| NED | [North, East, Down] | meters | Relative to earth's surface, useful for vizualizing. |
| Device | [Forward, Right, Down] | meters | This is the Mesh3D local frame. <br> Relative to camera, **not imu.** <br> ![img](http://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/RPY_angles_of_airplanes.png/440px-RPY_angles_of_airplanes.png)|
| Road | [Forward, Left, Up] | meters | On the road plane aligned to the vehicle. <br> ![img](https://upload.wikimedia.org/wikipedia/commons/f/f5/RPY_angles_of_cars.png) |
| View | [Right, Down, Forward] | meters | Like device frame, but according to camera conventions. |
| Camera | [u, v, focal] | pixels | Like view frame, but 2d on the camera image.|
| Normalized Camera | [u / focal, v / focal, 1] | / | |
| Model | [u, v, focal] | pixels | The sampled rectangle of the full camera frame the model uses. |
| Normalized Model | [u / focal, v / focal, 1] | / | |




Orientation Conventations
------
Quaternions, rotation matrices and euler angles are three
equivalent representations of orientation and all three are
used throughout the code base.

For euler angles the preferred convention is [roll, pitch, yaw]
which corresponds to rotations around the [x, y, z] axes. All
euler angles should always be in radians or radians/s unless
for plotting or display purposes. For quaternions the hamilton
notations is preferred which is [q<sub>w</sub>, q<sub>x</sub>, q<sub>y</sub>, q<sub>z</sub>]. All quaternions
should always be normalized with a strictly positive q<sub>w</sub>. **These
quaternions are a unique representation of orientation whereas euler angles
or rotation matrices are not.**

To rotate from one frame into another with euler angles the
convention is to rotate around roll, then pitch and then yaw,
while rotating around the rotated axes, not the original axes.


Calibration
------
EONs are not all mounted in the exact same way. To compensate for the effects of this the vision model takes in an image that is "calibrated". This means the image is aligned so the direction of travel of the car when it is going straight and the road is flat is always in the location on the image. This calibration is defined by a pitch and yaw angle that describe the direction of travel vector in device frame.

Example
------
To transform global Mesh3D positions and orientations (positions_ecef, quats_ecef) into the local frame described by the
first position and orientation from Mesh3D one would do:
```
ecef_from_local = rot_from_quat(quats_ecef[0])
local_from_ecef = ecef_from_local.T
positions_local = np.einsum('ij,kj->ki', local_from_ecef, postions_ecef - positions_ecef[0])
rotations_global = rot_from_quat(quats_ecef)
rotations_local = np.einsum('ij,kjl->kil', local_from_ecef, rotations_global)
eulers_local = euler_from_rot(rotations_local)
```
