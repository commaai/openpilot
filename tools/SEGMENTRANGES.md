## Segment Ranges

You can specify which segments from a route to load using the following syntax:

```plaintext
# The 4th segment
a2a0ccea32023010|2023-07-27--13-01-19/4

# The 4th, 5th, and 6th segments
a2a0ccea32023010|2023-07-27--13-01-19/4:6

# The last segment
a2a0ccea32023010|2023-07-27--13-01-19/-1

# The first 5 segments
a2a0ccea32023010|2023-07-27--13-01-19/:5

# All except the first segment
a2a0ccea32023010|2023-07-27--13-01-19/1:
