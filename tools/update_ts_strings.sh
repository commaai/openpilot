BASEDIR=~/openpilot/openpilot  # temp

# lupdate finds and adds all strings wrapped in tr() to main_languagecode files
lupdate $BASEDIR/selfdrive/ui/qt -ts $BASEDIR/selfdrive/ui/translations/main_en.ts $BASEDIR/selfdrive/ui/translations/main_fr.ts

# once translated, run
#lrelease $BASEDIR/selfdrive/ui/translations/main_en.ts $BASEDIR/selfdrive/ui/translations/main_fr_backup.ts
# which converts the translations to a binary file that enables fast lookups by the application
