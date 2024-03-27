#!/bin/bash -e

sudo openocd -f "interface/stlink.cfg" -c "transport select hla_swd" -f "target/stm32h7x.cfg" -c "init"
