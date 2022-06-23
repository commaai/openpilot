#!/bin/bash

lupdate -extensions cc,h -recursive '.' -ts translations/main_fr.ts translations/test_en.ts
# lupdate finds and adds all strings wrapped in tr() to main_languagecode files
linguist translations/main_fr.ts

# once translated, run
lrelease translations/*
# which converts the translations to a binary file that enables fast lookups by the application
