#!/bin/bash

echo "compressing training guide images"
optipng -o7 -strip all training/* training_wide/*

# This can sometimes provide smaller images
# mogrify -quality 100 -format jpg training_wide/* training/*
