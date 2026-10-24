#!/bin/sh
install_loc="$HOME/.local/bin"
docker build --platform=linux/arm64 -t cuda-nvcc:12.8 - <<'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-nvdisasm-12-8 cuda-cuobjdump-12-8 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:$PATH
EOF

mkdir -p "$install_loc"
tee "$install_loc/nvccshim" >/dev/null <<'EOF'
#!/bin/sh
set -eu
cname="cuda-nvcc-persistent"
if ! docker inspect --format='{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; then
  docker rm -f "$cname" 2>/dev/null || true
  docker run -d --platform=linux/arm64 --name "$cname" \
    -v /var/folders:/var/folders -v "$HOME":"$HOME" \
    cuda-nvcc:12.8 sleep 300 >/dev/null
fi
exec docker exec "$cname" "$(basename "$0")" "$@"
EOF
chmod +x "$install_loc/nvccshim"
for t in nvcc nvdisasm; do
  ln -sf "$install_loc/nvccshim" "$install_loc/$t"
done
