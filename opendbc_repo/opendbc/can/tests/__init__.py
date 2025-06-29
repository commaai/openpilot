import glob
import os

from opendbc import DBC_PATH

ALL_DBCS = [os.path.basename(dbc).split('.')[0] for dbc in
            glob.glob(f"{DBC_PATH}/*.dbc")]
TEST_DBC = os.path.abspath(os.path.join(os.path.dirname(__file__), "test.dbc"))
