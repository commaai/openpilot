#!/bin/bash

echo "compressing training guide images"
optipng -o7 -strip all training/* training_wide/*
