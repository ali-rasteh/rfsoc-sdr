#!/bin/bash
#
# Startup script for PynqLinux (Ubuntu 22.04 based)
# This script runs automatically at boot via systemd.
# /usr/local/sbin/startup_script.sh

set -euo pipefail

LOGFILE="/var/log/startup_script.log"
# Send all script output to the logfile
exec >>"$LOGFILE" 2>&1
echo "---------------------------------------"
echo "Boot at: $(date)"


# Configuring network
echo "Configuring network..."
/usr/sbin/sysctl -w net.ipv4.ip_forward=1
echo 1 | /usr/bin/tee /proc/sys/net/ipv4/ip_forward >/dev/null
/usr/sbin/ip addr add 192.168.185.4/24 dev eth0 || true
/usr/sbin/ip route replace default via 192.168.185.1 dev eth0
/usr/bin/resolvectl dns eth0 8.8.8.8

# # Start the Python program
# echo "Starting Python app..."
# # Replace the shell with python so systemd tracks it as the main process
# cd /home/xilinx/jupyter_notebooks/sounder_sdr
# exec /usr/local/share/pynq-venv/bin/python ./rfsoc_test.py


echo "Startup script finished at $(date)" >> "$LOGFILE"
