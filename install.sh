#!/bin/bash

# ==========================================
#   EzzPilotAR Installer for Comma Devices
#   Repo: github.com/aamabdulrhman-sudo/EzzPilotAR
# ==========================================
#
# طريقة التثبيت عبر SSH على جهاز Comma:
#   curl -sSL https://raw.githubusercontent.com/aamabdulrhman-sudo/EzzPilotAR/master/install.sh | bash
#
# أو للتثبيت بدون SSH (من الإعدادات):
#   Settings → Software → Custom Software
#   أدخل: https://github.com/aamabdulrhman-sudo/EzzPilotAR

set -e

REPO_URL="https://github.com/aamabdulrhman-sudo/EzzPilotAR.git"
BRANCH="${BRANCH:-master}"
INSTALL_DIR="/data/openpilot"
BACKUP_DIR="/data/openpilot.bak"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "============================================"
echo "    EzzPilotAR - Comma Device Installer"
echo "============================================"
echo -e "${NC}"

# 1. إيقاف openpilot
echo -e "${GREEN}[*] إيقاف openpilot...${NC}"
pkill -f manager.py 2>/dev/null || true
sleep 2

# 2. نسخ احتياطي للتثبيت الحالي
if [ -d "$INSTALL_DIR" ]; then
  echo -e "${GREEN}[*] نسخ احتياطي للتثبيت السابق...${NC}"
  rm -rf "$BACKUP_DIR"
  mv "$INSTALL_DIR" "$BACKUP_DIR"
fi

# 3. تحميل المشروع مع الـ submodules (opendbc, panda, ...)
echo -e "${GREEN}[*] تحميل EzzPilotAR من GitHub (مع submodules)...${NC}"
git clone --recurse-submodules "$REPO_URL" -b "$BRANCH" "$INSTALL_DIR"

cd "$INSTALL_DIR"

# 4. البناء (إذا لم يكن prebuilt)
if [ ! -f prebuilt ]; then
  echo -e "${GREEN}[*] بناء openpilot...${NC}"
  cd system/manager
  ./build.py || echo -e "${RED}[!] تحذير: قد تحتاج لإعادة البناء يدوياً${NC}"
  cd "$INSTALL_DIR"
fi

# 5. الانتهاء
echo -e "${CYAN}"
echo "============================================"
echo "         تم التثبيت بنجاح!"
echo "============================================"
echo -e "${NC}"
echo -e "${GREEN}[*] إعادة التشغيل لتطبيق التغييرات...${NC}"
sleep 3
reboot
