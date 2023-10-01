import os
import shlex

from openpilot.selfdrive.debug.auto_fingerprint import auto_fingerprint


FINGERPRINT_COMMAND = "/fingerprint"


if __name__ == "__main__":
  comment = os.environ["COMMENT_BODY"]

  for line in comment.split("\n"):
    if FINGERPRINT_COMMAND in line:
      start = line.index(FINGERPRINT_COMMAND)

      split = shlex.split(line[start:])

      if len(split) not in [2,3]:
        raise Exception(f"Invalid number of arguments: {split}")

      if len(split) == 2:
        _, route = split
        platform = None
      if len(split) == 3:
        _, route, platform = split

      platform = auto_fingerprint(route, platform)

      with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        print(f'pr-branch-name=fingerprint-{platform.replace(" ", "-").lower()}', file=fh)
        print(f'pr-title=Fingerprint: automatic fingerprint for {platform}', file=fh)
        print(f'pr-comment=Route: [{route}](https://useradmin.comma.ai/?onebox={route})', file=fh)

      exit(0)

  raise Exception(f"{FINGERPRINT_COMMAND} is required to call this job")
