#!/usr/bin/env bash

BRANCH="$(git branch --show-current)"
TOML_VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
LATEST_TAG_VERSION="$(git tag --list | sort -V -r | head -n 1)"
TAGGED_VERSION=""

if [[ "$BRANCH" != "master" ]]; then
  echo "Not on master branch."
  exit 1
fi

if [[ "$TOML_VERSION" == "$LATEST_TAG_VERSION" ]]; then
  TAGGED_VERSION=$(echo "$TOML_VERSION" | python3 -c "v = input().split('.'); v[-1]=str(int(v[-1])+1); print('.'.join(v))")
  sed -i "s/version = \"$TOML_VERSION\"/version = \"$TAGGED_VERSION\"/" pyproject.toml
elif [[ -z "$LATEST_TAG_VERSION" ]] || printf "$LATEST_TAG_VERSION\n$TOML_VERSION" | sort -V -C; then
  TAGGED_VERSION="$TOML_VERSION"
else
  echo "Version in pyproject.toml is lower than the latest tag version."
  exit 1
fi

echo "Tagging $TAGGED_VERSION..."
if [[ -n "$(git ls-files -m | grep pyproject.toml)" ]]; then
  echo "Commiting pyproject.toml..."
  git add pyproject.toml
  git commit --no-verify -m "Bump version to $TAGGED_VERSION"
fi
git tag "$TAGGED_VERSION"
