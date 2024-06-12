#!/bin/bash

echo "compressing training guide images"
optipng -o7 -strip all training/*

# This can sometimes provide smaller images
# mogrify -quality 100 -format jpg training/*
