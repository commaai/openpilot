#!/bin/sh

ssh -tt -o StrictHostKeyChecking=no -i $KEY_FILE comma@192.168.63.239 $@