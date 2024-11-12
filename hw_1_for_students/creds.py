import requests
import uuid
import json
import os
from dotenv import load_dotenv

ca_cert_path = 'sertificates/russian_trusted_root_ca.cer'
sb_auth_data = os.getenv('SB_AUTH_DATA')
load_dotenv('.env')

# Generate a UUID4
unique_id = str(uuid.uuid4())
url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

payload='scope=GIGACHAT_API_PERS'
headers = {
  'Content-Type': 'application/x-www-form-urlencoded',
  'Accept': 'application/json',
  'RqUID': unique_id,
  'Authorization': f'Basic {sb_auth_data}'
}

response = requests.request("POST", url, headers=headers, data=payload, verify=False)
response.text
# access_token = json.loads(response.text)['access_token']


# url = "https://gigachat.devices.sberbank.ru/api/v1/models"

# payload={}

# auth_token = f"Bearer {access_token}"
# headers = {
#   'Accept': 'application/json',
#   'Authorization': auth_token
# }

# response = requests.request("GET", url, headers=headers, data=payload, verify=False)

# print(response.text)