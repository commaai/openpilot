#!/bin/bash

cppcheck --force -j$(nproc) --output-file=cppcheck_out.txt \
  selfdrive/ common/ opendbc/ cereal/ installer/

