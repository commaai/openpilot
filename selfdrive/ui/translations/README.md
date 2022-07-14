# Multilanguage

![multilanguage_onroad](https://user-images.githubusercontent.com/25857203/178912800-2c798af8-78e3-498e-9e19-35906e0bafff.png)

## Contributing

Before getting started, make sure you have set up the openpilot Ubuntu development environment by reading the [tools README.md](https://github.com/commaai/openpilot/tree/master/tools).

### Adding a New Language

openpilot provides a few tools to help contributors manage their translations and to ensure quality. To get started:

1. Add your new language to [languages.json](https://github.com/commaai/openpilot/blob/master/selfdrive/ui/translations/languages.json) with the appropriate [language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) and the localized language name (Simplified Chinese is `中文（繁體）`).
2. Run `selfdrive/ui/update_translations.py` to generate the translation file (`*.ts`).
3. Run `linguist selfdrive/ui/translations/your_language_file.ts` to edit the translation file, marking each translation as completed.
4. Save your file and run `selfdrive/ui/update_translations.py --release` to generate the compiled QM file used by the Qt UI.

### Improving an Existing Language

Follow steps above, ommitting steps 1. and 2.
