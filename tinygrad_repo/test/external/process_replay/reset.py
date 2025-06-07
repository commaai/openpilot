#!/usr/bin/env python3
from tinygrad.helpers import db_connection, VERSION
cur = db_connection()
cur.execute(f"drop table if exists kernel_process_replay_{VERSION}")
cur.execute(f"drop table if exists schedule_process_replay_{VERSION}")
