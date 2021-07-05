#!/bin/bash

echo "compressing training guide images"
optipng -o7 -strip all training/*
optipng -o7 -strip all training_wide/*
