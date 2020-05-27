#!/bin/bash

cppcheck --force -j$(nproc) \
  selfdrive/ common/ opendbc/ cereal/ installer/

