TEST_TEXT="SOURCE_TEXT_WRAPPED"

lupdate -extensions cc,h -recursive '.' -ts translations/main_test_en.ts
sed -i 's/<translation type="unfinished"><\/translation>/<translation>'"$TEST_TEXT"'<\/translation>/' translations/main_test_en.ts
lrelease translations/main_test_en.ts
