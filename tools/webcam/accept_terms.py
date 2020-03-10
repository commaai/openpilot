#!/usr/bin/env python
from common.params import Params
from selfdrive.version import terms_version

if __name__ == '__main__':
  params = Params()
  params.put("HasAcceptedTerms", str(terms_version, 'utf-8'))
  print("Terms Accepted!")