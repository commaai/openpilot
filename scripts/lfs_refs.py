#!/usr/bin/env python3
import argparse
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


def pointers(lfs, *refs):
  files = json.loads(subprocess.check_output([lfs, 'ls-files', '--json', '--include=', '--exclude=', *refs], text=True))['files'] or []
  return {f['oid']: f"version {f['version']}\noid {f['oid_type']}:{f['oid']}\nsize {f['size']}\n" for f in files}


def archive(repo, objects, check=False):
  credentials = subprocess.run(['git', 'config', '--null', '--get-regexp', r'^credential\.'], text=True, stdout=subprocess.PIPE).stdout
  env = dict(os.environ)
  for key in git('rev-parse', '--local-env-vars').splitlines():
    env.pop(key, None)
  if not check:
    for role in ('AUTHOR', 'COMMITTER'):
      identity = git('var', f'GIT_{role}_IDENT').rsplit(' ', 2)[0]
      name, email = identity.rsplit(' <', 1)
      env[f'GIT_{role}_NAME'], env[f'GIT_{role}_EMAIL'] = name, email[:-1]

  with tempfile.TemporaryDirectory(prefix='lfs-refs-') as directory:
    def run(*args, **kwargs):
      return git('-C', directory, '-c', 'core.hooksPath=/dev/null', *args, env=env, **kwargs)

    run('init', '--bare', '--quiet')
    for entry in credentials.split('\0'):
      if '\n' in entry:
        key, value = entry.split('\n', 1)
        run('config', '--add', key, value)
    ref = 'refs/heads/lfs-history'
    for attempt in range(3):
      parent = None
      if run('ls-remote', repo, ref):
        run('fetch', '--quiet', '--depth=1', repo, ref)
        parent = run('rev-parse', 'FETCH_HEAD')
        run('read-tree', parent)
      else:
        run('read-tree', '--empty')

      entries = {line.split('\t')[1]: line.split()[1] for line in run('ls-files', '--stage').splitlines()}
      additions = {}
      for oid, pointer in objects.items():
        path = f'objects/{oid[:2]}/{oid}'
        blob = run('hash-object', '-w', '--stdin', input=pointer)
        if path in entries:
          if entries[path] != blob:
            raise RuntimeError(f'Incorrect HF LFS reference: {path}')
        else:
          additions[path] = blob
      if not additions:
        return
      if check:
        raise RuntimeError(f"Missing HF LFS references: {', '.join(additions)}. Run scripts/lfs_refs.py --sync <ref> with HF write access.")

      if not parent:
        additions['.gitattributes'] = run('hash-object', '-w', '--stdin', input='objects/** filter=lfs diff=lfs merge=lfs -text\n')
      run('update-index', '--index-info', input=''.join(f'100644 {blob}\t{path}\n' for path, blob in additions.items()))
      attributes = run('check-attr', '--cached', 'filter', '--stdin', input='\n'.join(path for path in additions if path.startswith('objects/')))
      if any(not line.endswith(': filter: lfs') for line in attributes.splitlines()):
        raise RuntimeError('HF lfs-history must track objects/** with Git LFS')
      tree = run('write-tree')
      commit = run('commit-tree', tree, *(['-p', parent] if parent else []), '-m', 'Retain openpilot LFS objects')
      try:
        run('push', '--quiet', repo, f'{commit}:{ref}')
        return
      except subprocess.CalledProcessError:
        current = run('ls-remote', repo, ref).split()
        if attempt == 2 or (current[0] if current else None) == parent:
          raise


def main():
  parser = argparse.ArgumentParser()
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument('--check', action='store_true')
  mode.add_argument('--sync', action='store_true')
  parser.add_argument('--lfs', default='git-lfs')
  parser.add_argument('refs', nargs='*')
  args = parser.parse_args()
  if not (args.check or args.sync) and not args.refs:
    parser.error('expected pre-push remote arguments, --check, or --sync')
  repo = hf_repo('origin' if args.check or args.sync else args.refs[0])
  if args.check or args.sync:
    if repo:
      objects = pointers(args.lfs, *args.refs)
      if args.sync and objects:
        subprocess.run([args.lfs, 'push', '--object-id', 'origin', *objects], check=True)
      archive(repo, objects, check=args.check)
    return

  updates = sys.stdin.read()
  command = [args.lfs, 'pre-push', *args.refs]
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
      objects.update(pointers(args.lfs, remote, local))
  archive(repo, {oid: objects[oid] for oid in oids})


if __name__ == '__main__':
  try:
    main()
  except (RuntimeError, subprocess.CalledProcessError) as e:
    sys.exit(f'LFS reference update failed: {e}')
