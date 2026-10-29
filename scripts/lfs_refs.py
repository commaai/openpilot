#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def git(*args, **kwargs):
  return subprocess.run(['git', *args], check=True, text=True, stdout=subprocess.PIPE, **kwargs).stdout.strip()


def hf_repo(remote):
  config = {}
  if Path('.lfsconfig').is_file():
    config.update(entry.split('\n', 1) for entry in git('config', '--file', '.lfsconfig', '--null', '--list').split('\0') if '\n' in entry)
  config.update(entry.split('\n', 1) for entry in git('config', '--null', '--list').split('\0') if '\n' in entry)
  url = next((config[key] for key in ('lfs.pushurl', 'lfs.url', f'remote.{remote}.lfspushurl', f'remote.{remote}.lfsurl') if key in config), '')
  return url.removesuffix('/info/lfs') if re.fullmatch(r'https://huggingface\.co/(?:datasets/)?[^/]+/[^/]+\.git/info/lfs', url) else None


def archive(repo, objects):
  with tempfile.TemporaryDirectory(prefix='lfs-refs-') as directory:
    ref = f'refs/lfs-archive/{Path(directory).name}'
    def run(*args, **kwargs):
      return git(*args, env={**os.environ, 'GIT_INDEX_FILE': f'{directory}/index'}, **kwargs)

    try:
      for _ in range(3):
        git('fetch', '--quiet', '--no-tags', '--no-write-fetch-head', repo, f'refs/heads/lfs-history:{ref}')
        parent = git('rev-parse', ref)
        run('read-tree', parent)
        entries = []
        for oid, pointer in objects.items():
          blob = git('hash-object', '-w', '--stdin', input=pointer)
          entries.append(f'100644 {blob}\tobjects/{oid[:2]}/{oid}\n')
        run('update-index', '--index-info', input=''.join(entries))
        tree = run('write-tree')
        if tree == git('rev-parse', f'{parent}^{{tree}}'):
          return
        commit = git('commit-tree', tree, '-p', parent, '-m', 'Retain openpilot LFS objects')
        if subprocess.run(['git', 'push', '--quiet', '--no-verify', repo, f'{commit}:refs/heads/lfs-history']).returncode == 0:
          return
      raise RuntimeError('Could not update HF lfs-history; retry the push')
    finally:
      git('update-ref', '-d', ref)


def main():
  lfs, *args = sys.argv[1:]
  repo = hf_repo(args[0])
  updates = sys.stdin.read()
  command = [lfs, 'pre-push', *args]
  if not repo or os.environ.get('GIT_LFS_SKIP_PUSH', '').lower() in ('1', 'true'):
    subprocess.run(command, input=updates, text=True, check=True)
    return
  pending = subprocess.check_output([*command, '--dry-run'], input=updates, text=True)
  oids = {line.split()[1] for line in pending.splitlines() if re.match(r'\S+ [0-9a-f]{64} => ', line)}
  subprocess.run(command, input=updates, text=True, check=True)
  if not oids:
    return

  objects = {}
  for update in updates.splitlines():
    _, local, _, remote = update.split()
    if set(local) != {'0'}:
      if set(remote) == {'0'}:
        remote = git('hash-object', '-w', '-t', 'tree', '--stdin', input='')
      files = json.loads(subprocess.check_output([lfs, 'ls-files', '--json', '--include=', '--exclude=', remote, local], text=True))['files'] or []
      objects.update({f['oid']: f"version {f['version']}\noid {f['oid_type']}:{f['oid']}\nsize {f['size']}\n" for f in files})
  archive(repo, {oid: objects[oid] for oid in oids})


if __name__ == '__main__':
  try:
    main()
  except (RuntimeError, subprocess.CalledProcessError) as e:
    sys.exit(f'LFS reference update failed: {e}')
