#!/usr/bin/env bash
sphinx-apidoc -o source ../ \
  ../xx ../laika_repo ../rednose_repo ../pyextra ../notebooks ../panda_jungle \
  ../third_party \
  ../panda/examples \
  ../scripts \
  ../selfdrive/modeld \
  ../selfdrive/debug \
  $(find .. -type d -name "*test*")
