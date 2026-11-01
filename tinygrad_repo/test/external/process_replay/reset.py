#!/usr/bin/env python3
from tinygrad.helpers import db_connection, VERSION
cur = db_connection()
for table in ("process_replay", "emu_process_replay"):
  cur.execute(f"drop table if exists {table}_{VERSION}")
