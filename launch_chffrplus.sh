#!/usr/bin/bash

# TODO: this can be removed after 0.9.6 release

# migrate continue.sh and relaunch
cat << EOF > /data/continue.sh
#!/usr/bin/bash

cd /data/openpilot
exec ./launch_openpilot.sh
EOF

/data/continue.sh
