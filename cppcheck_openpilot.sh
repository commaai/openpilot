#!/bin/bash

cppcheck --force -j$(nproc) selfdrive/ common/ opendbc/ cereal/ installer/ 2> cppcheck_report.txt

