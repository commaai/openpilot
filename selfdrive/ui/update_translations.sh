#!/bin/bash

lupdate ui.pro -noobsolete
# lupdate finds and adds all strings wrapped in tr() to main_languagecode files
linguist translations/main_fr.ts

# once translated, run
lrelease ui.pro
# which converts the translations to a binary file that enables fast lookups by the application
