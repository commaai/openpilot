from common.api import *
from common.params import *
a=Api(Params().get('DongleId', encoding='utf8'))
token = a.get_token()
print(a.dongle_id)
print(token)
site = 'v1.1/devices/' + a.dongle_id + '/stats'
print(site)
txt = api_get(site, access_token=token).text
print(txt)

