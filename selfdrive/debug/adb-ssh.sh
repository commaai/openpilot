#!/bin/bash

adb forward tcp:8022 tcp:8022
ssh localhost -p 8022 -i ~/openpilot/xx/phone/key/id_rsa
