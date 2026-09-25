#!/bin/sh
# Grant passwordless sudo for exactly the two root-only probes localbench uses:
#   powermetrics  - GPU/CPU/ANE power, GPU frequency, thermal pressure
#   purge         - drop the file cache so "cold load" timings are real
# Nothing else. Run once, interactively:  sudo ./scripts/install-sudoers.sh
# Remove with:                             sudo rm /etc/sudoers.d/localbench
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo: sudo $0" >&2
    exit 1
fi

target_user="${SUDO_USER:?run via sudo so SUDO_USER is set}"
dest=/etc/sudoers.d/localbench
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

printf '%s ALL=(root) NOPASSWD: /usr/bin/powermetrics, /usr/sbin/purge\n' "$target_user" >"$tmp"
visudo -cf "$tmp" >/dev/null
install -m 0440 -o root -g wheel "$tmp" "$dest"
echo "installed $dest:"
cat "$dest"
