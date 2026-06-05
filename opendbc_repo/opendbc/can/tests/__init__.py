import glob
import os

from opendbc import DBC_PATH, get_generated_dbcs

static_dbcs = [os.path.basename(dbc).split('.')[0] for dbc in
               glob.glob(f"{DBC_PATH}/*.dbc")]
ALL_DBCS = sorted(set(static_dbcs + list(get_generated_dbcs().keys())))
TEST_DBC = os.path.abspath(os.path.join(os.path.dirname(__file__), "test.dbc"))
