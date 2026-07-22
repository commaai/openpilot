#!/usr/bin/env python3
import openpilot.cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog


def main():
  pm = messaging.PubMaster(['userBookmark'])
  sm = messaging.SubMaster(['bookmarkButton'])

  while True:
    sm.update()

    if sm.updated['bookmarkButton']:
      cloudlog.info("Bookmark button pressed!")
      msg = messaging.new_message('userBookmark', valid=True)
      pm.send('userBookmark', msg)


if __name__ == '__main__':
  main()
