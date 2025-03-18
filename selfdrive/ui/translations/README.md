# Multilanguage

[![languages](https://raw.githubusercontent.com/commaai/openpilot/badges/translation_badge.svg)](#)

## Contributing

Before getting started, make sure you have set up the openpilot Ubuntu development environment by reading the [tools README.md](/tools/README.md).

### Policy

Most of the languages supported by openpilot come from and are maintained by the community via pull requests. A pull request likely to be merged is one that [fixes a translation or adds missing translations.](https://github.com/commaai/openpilot/blob/master/selfdrive/ui/translations/README.md#improving-an-existing-language)

We also generally merge pull requests adding support for a new language if there are community members willing to maintain it. Maintaining a language is ensuring quality and completion of translations before each openpilot release.

comma may remove or hide language support from releases depending on translation quality and completeness.

### Adding a New Language

openpilot provides a few tools to help contributors manage their translations and to ensure quality. To get started:

1. Add your new language to [languages.json](/selfdrive/ui/translations/languages.json) with the appropriate [language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) and the localized language name (Traditional Chinese is `中文（繁體）`).
2. Generate the XML translation file (`*.ts`):
   ```shell
   selfdrive/ui/update_translations.py
   ```
3. Edit the translation file, marking each translation as completed:
   ```shell
   linguist selfdrive/ui/translations/your_language_file.ts
   ```
4. View your finished translations by compiling and starting the UI, then find it in the language selector:
   ```shell
   scons -j$(nproc) selfdrive/ui && selfdrive/ui/ui
   ```
5. Read [Checking the UI](#checking-the-ui) to double-check your translations fit in the UI.

### Improving an Existing Language

Follow step 3. above, you can review existing translations and add missing ones. Once you're done, just open a pull request to openpilot.

### Checking the UI
Different languages use varying space to convey the same message, so it's a good idea to double-check that your translations do not overlap and fit into each widget. Start the UI (step 4. above) and view each page, making adjustments to translations as needed.

#### To view offroad alerts:

With the UI started, you can view the offroad alerts with:
```shell
selfdrive/ui/tests/cycle_offroad_alerts.py
```

### Updating the UI

Any time you edit source code in the UI, you need to update the translations to ensure the line numbers and contexts are up to date (first step above).

### Testing

openpilot has a few unit tests to make sure all translations are up-to-date and that all strings are wrapped in a translation marker. They are run in CI, but you can also run them locally.

Tests translation files up to date:

```shell
selfdrive/ui/tests/test_translations.py
```

Tests all static source strings are wrapped:

```shell
selfdrive/ui/tests/create_test_translations.sh && selfdrive/ui/tests/test_translations
```

---
![multilanguage_onroad](https://user-images.githubusercontent.com/25857203/178912800-2c798af8-78e3-498e-9e19-35906e0bafff.png)
