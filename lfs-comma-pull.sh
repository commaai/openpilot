#!/usr/bin/env bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
FETCH_ALL=0
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -a|--all) FETCH_ALL=1 ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if git-lfs is installed and initialized
if ! command -v git-lfs >/dev/null 2>&1; then
  echo -e "${RED}Error: git-lfs is not installed${NC}"
  echo "Please install git-lfs first:"
  echo "  brew install git-lfs    # On macOS"
  echo "  apt-get install git-lfs # On Ubuntu/Debian"
  exit 1
fi

# Function to get LFS objects that need syncing
get_missing_lfs_objects() {
  git lfs ls-files | awk '$2 == "-" {$1=""; $2=""; sub(/^[ \t]+/, ""); print}'
}

# Function to get all LFS objects
get_all_lfs_objects() {
  git lfs ls-files | awk '{$1=""; $2=""; sub(/^[ \t]+/, ""); print}'
}

# Clean up any existing LFS config
echo -e "${YELLOW}Cleaning up existing LFS configurations...${NC}"
git config --unset-all lfs.url
git config --unset-all lfs.pushurl
git config --local --unset-all lfs.url
git config --local --unset-all lfs.pushurl

# Backup current LFS config
if [ -f .lfsconfig ]; then
  cp .lfsconfig .lfsconfig.backup
fi

# Switch to upstream config and pull missing objects
echo -e "${YELLOW}Using upstream LFS configuration...${NC}"
if [ -f .lfsconfig-comma ]; then
  cp .lfsconfig-comma .lfsconfig
  echo -e "${GREEN}Switched to upstream LFS configuration${NC}"

  if [ $FETCH_ALL -eq 1 ]; then
    echo -e "${YELLOW}Fetching all LFS objects (including already present)...${NC}"

    git lfs fetch --all || {
      echo -e "${RED}Failed to fetch LFS objects${NC}"
      if [ -f .lfsconfig.backup ]; then
        mv .lfsconfig.backup .lfsconfig
      fi
      exit 1
    }

    git lfs checkout || {
      echo -e "${RED}Failed to checkout LFS objects${NC}"
      if [ -f .lfsconfig.backup ]; then
        mv .lfsconfig.backup .lfsconfig
      fi
      exit 1
    }

    echo -e "${GREEN}Successfully fetched all LFS objects${NC}"
  else
    # Get list of missing LFS objects
    MISSING_FILES=$(get_missing_lfs_objects)

    if [ ! -z "$MISSING_FILES" ]; then
      echo -e "${YELLOW}Missing files to fetch:${NC}"
      echo "$MISSING_FILES"

      git lfs fetch || {
        echo -e "${RED}Failed to fetch LFS objects${NC}"
        if [ -f .lfsconfig.backup ]; then
          mv .lfsconfig.backup .lfsconfig
        fi
        exit 1
      }

      git lfs checkout || {
        echo -e "${RED}Failed to checkout LFS objects${NC}"
        if [ -f .lfsconfig.backup ]; then
          mv .lfsconfig.backup .lfsconfig
        fi
        exit 1
      }

      echo -e "${GREEN}Successfully fetched missing LFS objects${NC}"
    else
      echo -e "${GREEN}No missing LFS objects to fetch${NC}"
    fi
  fi
else
  echo -e "${RED}Warning: .lfsconfig-comma not found${NC}"
fi

# Restore original config
if [ -f .lfsconfig.backup ]; then
  mv .lfsconfig.backup .lfsconfig
  echo -e "${GREEN}Restored original LFS configuration${NC}"
else
  rm -f .lfsconfig
fi

echo -e "${GREEN}LFS sync completed successfully${NC}"

# Continue with the commit
exit 0
