What is system?
----

In here we have daemons that have nothing to do with self driving.

A camera or an accelerometer is a generic service that can be used for many applications.

openpilot is a more opinionated ROS, and strives to be the operating system for life.

Relationship with selfdrive
----

While selfdrive can import from system, system can't import from selfdrive. selfdrive is an application that uses system.

There's a few violations of this rule that should be cleaned up.
