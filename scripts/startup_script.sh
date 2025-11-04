#!/bin/bash
#
# Sstartup script for PynqLinux (Ubuntu 22.04 based)
# This script runs automatically at boot via systemd.
#

LOGFILE="/var/log/startup_script.log"

echo "---------------------------------------" >> "$LOGFILE"
echo "Boot at: $(date)" >> "$LOGFILE"


# Set the IP addresses
echo "Configuring network..." >> "$LOGFILE"
sudo sysctl -w net.ipv4.ip_forward=1
sudo echo 1 > /proc/sys/net/ipv4/ip_forward
sudo ip addr add 192.168.185.4/24 dev eth0
sudo route add default gw 192.168.185.1 dev eth0
# sudo route add default gw 192.168.2.1 dev eth0
# sudo route add default gw 192.168.3.100 dev usb0
sudo resolvectl dns eth0 8.8.8.8
# sudo resolvectl dns usb0 8.8.8.8

# # Wait until network is ready
# echo "Waiting for network..." >> "$LOGFILE"
# until ping -c1 8.8.8.8 &>/dev/null; do
#     sleep 2
# done
# echo "Network is up at $(date)" >> "$LOGFILE"

# Start the Python program
echo "Starting Python app..." >> "$LOGFILE"
echo "Python version: $(python --version)" >> "$LOGFILE"
cd /home/xilinx/jupyter_notebooks/sounder_sdr
python ./rfsoc_test.py >> "$LOGFILE" 2>&1 &


echo "Startup script finished at $(date)" >> "$LOGFILE"
