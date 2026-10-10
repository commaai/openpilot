#!/bin/sh
install_loc="$HOME/.local/bin"
docker pull --platform=linux/amd64 rocm/dev-ubuntu-22.04:7.1.1
docker tag rocm/dev-ubuntu-22.04:7.1.1 rocm-hipcc:7.1.1

mkdir -p "$install_loc"
tee "$install_loc/hipccshim" >/dev/null <<'EOF'
#!/bin/sh
set -eu
cname="rocm-hipcc-persistent"
if ! docker inspect --format='{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; then
  docker rm -f "$cname" 2>/dev/null || true
  docker run -d --platform=linux/amd64 --name "$cname" \
    -v /var/folders:/var/folders -v "$HOME":"$HOME" \
    rocm-hipcc:7.1.1 sleep 300 >/dev/null
fi
exec docker exec "$cname" "$(basename "$0")" "$@"
EOF
chmod +x "$install_loc/hipccshim"
for t in hipcc hipconfig; do
  ln -sf "$install_loc/hipccshim" "$install_loc/$t"
done
