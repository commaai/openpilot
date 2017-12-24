#!/bin/bash
PYTHONPATH="." nosetests -x -s tests/automated/$1*.py

