import unittest
import os

from common.api import Api
from common.file_helpers import mkdirs_exists_ok
from common.basedir import PERSIST

class TestApiGetToken(unittest.TestCase):
	def is_private_key_generated(self):
		mkdirs_exists_ok(PERSIST+"/comma")
		assert os.system("openssl genrsa -out "+PERSIST+"/comma/id_rsa.tmp 2048") == 0
		assert os.system("openssl rsa -in "+PERSIST+"/comma/id_rsa.tmp -pubout -out "+PERSIST+"/comma/id_rsa.tmp.pub") == 0
		os.rename(PERSIST+"/comma/id_rsa.tmp", PERSIST+"/comma/id_rsa")
		os.rename(PERSIST+"/comma/id_rsa.tmp.pub", PERSIST+"/comma/id_rsa.pub")
	
	def test_get_token(self):
		# This is prerequisite to create jwt token
		self.is_private_key_generated()
		
		# Create Api class object with dummy DongleId
		test_obj = Api("cb38263377b873ee")
		# Obtain a jwt token
		test_token = test_obj.get_token()	
		
		# Check if the token is a string
		self.assertEqual(isinstance(test_token, str), True)
		
if __name__ == "__main__":
	unittest.main()
