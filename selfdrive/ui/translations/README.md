# Multilanguage

![multilanguage_onroad](https://user-images.githubusercontent.com/25857203/178912800-2c798af8-78e3-498e-9e19-35906e0bafff.png)

## Contributing

Before getting started, make sure you have set up the openpilot Ubuntu development environment by reading the [tools README.md](/tools/README.md).

### Adding a New Language

openpilot provides a few tools to help contributors manage their translations and to ensure quality. To get started:

1. Add your new language to [languages.json](/selfdrive/ui/translations/languages.json) with the appropriate [language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) and the localized language name (Simplified Chinese is `中文（繁體）`).
2. Generate the translation file (`*.ts`):
   ```shell
   selfdrive/ui/update_translations.py
   ```
3. Edit the translation file, marking each translation as completed:
   ```shell
   linguist selfdrive/ui/translations/your_language_file.ts
   ```
4. Save your file and generate the compiled QM file used by the Qt UI:
   ```shell
   selfdrive/ui/update_translations.py --release
   ```

### Improving an Existing Language

Follow the steps above, omitting steps 1. and 2. Any time you edit translations you'll want to make sure to compile them.

### Testing

openpilot has a unit test to make sure all translations are up to date with the text in openpilot and that all translations are completed.

Run and fix any issues:

```python
selfdrive/ui/tests/test_translations.py
```
