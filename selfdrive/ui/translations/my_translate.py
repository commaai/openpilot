#!/usr/bin/env python3

from auto_translate import get_language_files

def main():
  print('foo')
  temp = get_language_files("pt-BR")
  temp = get_language_files()
  print(list(temp))

  for lang, path in temp.items():
    print(f"Translate {lang} ({path})")
    # translate_file(path, lang, args.all_translations)

if __name__ == "__main__":
  main()

