#!/usr/bin/env python3
from tinygrad.helpers import db_connection, VERSION
cur = db_connection()
cur.execute(f"drop table if exists process_replay_{VERSION}")
