#!/usr/bin/env bash
set -euo pipefail

# Copy a local file or directory through the already established reverse SSH
# tunnel.  The NAS SFTP endpoint rejects rsync's remote-server mode, so this
# uses SCP/SFTP while retaining the same atomic temporary-path protocol. No
# deletion is requested; the remote destination is published with a final
# rename after the transfer. Credentials stay in SSH key files.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LOCAL_SOURCE REMOTE_RELATIVE_DESTINATION" >&2
  exit 64
fi

source_path=$1
relative_dest=$2
if [[ "$relative_dest" == /* || "$relative_dest" == *..* ]]; then
  echo "nas_sync: destination must be relative to NAS_PROJECT_ROOT" >&2
  exit 64
fi
if [[ ! -e "$source_path" ]]; then
  echo "nas_sync: source does not exist: $source_path" >&2
  exit 66
fi

: "${NAS_TUNNEL_HOST:=127.0.0.1}"
: "${NAS_TUNNEL_PORT:=24557}"
: "${NAS_USER:=Valadmin}"
: "${NAS_PROJECT_ROOT:=/home/A100/FGREF_RMOT_SPORTS_20261019}"
: "${NAS_IDENTITY_FILE:=/data3/liuyeqiang/.ssh/id_ed25519_nas_readonly}"
: "${NAS_KNOWN_HOSTS:=/data3/liuyeqiang/.ssh/known_hosts_nas_readonly}"

sftp_opts=(-P "$NAS_TUNNEL_PORT" -i "$NAS_IDENTITY_FILE" -o IdentitiesOnly=yes
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=nas-readonly
  -o UserKnownHostsFile="$NAS_KNOWN_HOSTS" -o ConnectTimeout=15)
remote="${NAS_PROJECT_ROOT}/${relative_dest}"
remote_dir=$(dirname "$remote")
base=$(basename "$remote")
token=".${base}.partial.$$.${RANDOM}"
remote_tmp="${remote_dir}/${token}"
remote_item="${remote_tmp}/${base}"

# The project root and its planned subdirectories already exist, but create
# each path component idempotently so a new report subdirectory is safe too.
mkdir_batch=""
path=""
IFS=/ read -r -a parts <<< "${remote_dir#/}"
for part in "${parts[@]}"; do
  path="${path}/${part}"
  mkdir_batch+="-mkdir ${path}\n"
done
printf '%b' "${mkdir_batch}mkdir ${remote_tmp}\nquit\n" | \
  sftp "${sftp_opts[@]}" "${NAS_USER}@${NAS_TUNNEL_HOST}" >/dev/null 2>&1

cleanup() {
  printf 'rm %s\nrmdir %s\nquit\n' "$remote_item" "$remote_tmp" | \
    sftp "${sftp_opts[@]}" "${NAS_USER}@${NAS_TUNNEL_HOST}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

scp_opts=(-P "$NAS_TUNNEL_PORT" -i "$NAS_IDENTITY_FILE" -o IdentitiesOnly=yes
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=nas-readonly
  -o UserKnownHostsFile="$NAS_KNOWN_HOSTS" -o ConnectTimeout=15)
if [[ -d "$source_path" ]]; then
  scp "${scp_opts[@]}" -r "$source_path" "${NAS_USER}@${NAS_TUNNEL_HOST}:${remote_item}"
else
  scp "${scp_opts[@]}" "$source_path" "${NAS_USER}@${NAS_TUNNEL_HOST}:${remote_item}"
fi

printf 'rename %s %s\nrmdir %s\nquit\n' "$remote_item" "$remote" "$remote_tmp" | \
  sftp "${sftp_opts[@]}" "${NAS_USER}@${NAS_TUNNEL_HOST}" >/dev/null
trap - EXIT

# Translate the SFTP path to the login-shell path for an end-to-end hash check.
shell_remote=${remote/#\/home\/A100/\/home\/Valadmin\/A100}
ssh "-p" "$NAS_TUNNEL_PORT" -i "$NAS_IDENTITY_FILE" -o IdentitiesOnly=yes \
  -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=nas-readonly \
  -o UserKnownHostsFile="$NAS_KNOWN_HOSTS" -o ConnectTimeout=15 \
  "${NAS_USER}@${NAS_TUNNEL_HOST}" "if test -f '$shell_remote'; then sha256sum -- '$shell_remote'; else stat -c '%s %n' '$shell_remote'; fi"
echo "nas_sync: published ${remote}"
