# DO NOT PUSH!
import requests

token = ''

response = requests.post("https://athena.comma.ai/0cf0840158be859e",
                         json={'method': 'getMessage', "params": {"service": "thermal", "timeout": 5000},
                               "jsonrpc": "2.0", "id": 0},
                         headers={'Authorization': f'JWT {token}'})

print(response)
print(response.json())
print(response.headers)
# curl https://athena.comma.ai/0cf0840158be859e \
# -d '{"method":"getMessage","params":{"service":"thermal","timeout":5000},"jsonrpc":"2.0","id":0}' \
# -H 'Authorization: JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2NTM1MDQ0ODYsIm5iZiI6MTYyMTk2ODQ4NiwiaWF0IjoxNjIxOTY4NDg2LCJpZGVudGl0eSI6IjA5MTYwNTNjOGJlZDg3ZDEifQ.TAoajbMBcSy_a9gQa36Pp_RbnvHvgL504-2LJ1ncTvA'
