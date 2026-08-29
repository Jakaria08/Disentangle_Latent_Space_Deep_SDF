#!/usr/bin/env bash
# Make /mnt/bulk10tb mount automatically at boot.
#
# Run with:  sudo bash add_bulk10tb_fstab.sh
#
# Safe by construction: backs up /etc/fstab, refuses to add a duplicate entry, and validates
# the new entry with `mount -a` before leaving. Uses UUID rather than /dev/sda because device
# letters can change between boots. `nofail` means a missing/failed disk cannot block boot.
set -euo pipefail

UUID="5933C2372974E040"
MNT="/mnt/bulk10tb"
LINE="UUID=${UUID}  ${MNT}  ntfs-3g  rw,noatime,allow_other,default_permissions,nofail,x-systemd.device-timeout=10  0  0"

[[ $EUID -eq 0 ]] || { echo "must run as root: sudo bash $0"; exit 1; }

if grep -qs "${UUID}\|[[:space:]]${MNT}[[:space:]]" /etc/fstab; then
  echo "An entry for ${MNT} or that UUID already exists in /etc/fstab:"
  grep -n "${UUID}\|${MNT}" /etc/fstab
  echo "Nothing changed."
  exit 0
fi

BACKUP="/etc/fstab.bak.$(date +%Y%m%d-%H%M%S)"
cp -a /etc/fstab "$BACKUP"
echo "backed up /etc/fstab -> $BACKUP"

mkdir -p "$MNT"
printf '\n# 10TB bulk data disk (Deep3DComp results)\n%s\n' "$LINE" >> /etc/fstab
echo "added:"
echo "  $LINE"

echo "validating with 'mount -a' ..."
if mount -a; then
  findmnt -no SOURCE,TARGET,FSTYPE,OPTIONS "$MNT" && echo "OK - entry is valid and will mount at boot."
else
  echo "mount -a FAILED - restoring $BACKUP so boot is not affected."
  cp -a "$BACKUP" /etc/fstab
  exit 1
fi
