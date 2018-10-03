#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo of the Google Assistant GRPC recognizer."""

import logging
import json
import aiy.assistant.grpc
import aiy.audio
import aiy.voicehat
from datetime import datetime
from pytz import timezone
import httplib2
import urllib.parse
import urllib.request
import oauth2client
from oauth2client.file import Storage
from googleapiclient.discovery import build
import sys
import requests
import nltk
from nltk import word_tokenize
from nltk import pos_tag

# "logged" contains all the nutrients that are logged
logged = ["Energy", "Protein", "Total lipid (fat)", "Carbohydrate, by difference", "Fiber, total dietary", "Sugars, total", "Sugars, added", "Calcium, Ca", "Iron, Fe", "Sodium, Na", "Vitamin C, total ascorbic acid", "Vitamin A, IU", "Fatty acids, total saturated", "Fatty acids, total trans", "Cholesterol"]
# "allergens" contains all the possible allergens
allergens = ["Milk", "Eggs", "Fish", "Crustacean shellfish", "Tree nuts", "Peanuts", "Wheat", "Soybeans"]
# "allergens_list" stores the list of allergens that the food contains
allergens_list = []

def sheet_oauth():
    '''
       This method authorizes the sheets API requests. The client_id , client_secret, refresh_token
        are obtained by allowing the app from a browser request.
       Returns credentials to make API requests to google sheet.
    '''
    client_id = "759306969044-tnqihvtr1cm0us78g81iv9th3ohseg4v.apps.googleusercontent.com"
    client_secret = "2OgTLPBlYk5t_HkpDzLyNpmD"
    refresh_token = "1/qZ5F4YdLn8cXQECg-SiLGEnwBwrWgvbQpC1a81krkHo"
    request = urllib.request.Request('https://accounts.google.com/o/oauth2/token',
    data=urllib.parse.urlencode({
      'grant_type':    'refresh_token',
      'client_id':     client_id,
      'client_secret': client_secret,
      'refresh_token': refresh_token
    }).encode("utf-8"),
    headers={
      'Content-Type': 'application/x-www-form-urlencoded',
      'Accept': 'application/json'
    }
    )
    with urllib.request.urlopen(request) as f:
         resp = f.read()
    access_token = json.loads(resp.decode("utf-8"))['access_token']
    credentials = oauth2client.client.AccessTokenCredentials(
                                    access_token=access_token,
                                    user_agent="Google sheets API Creds",
                                    revoke_uri="https://accounts.google.com/o/oauth2/revoke")
    return credentials

def write_to_sheet(credentials,spreadsheetId, input):
    '''
       This method appends the input onto the given spreadsheet.
    '''
    http = credentials.authorize(httplib2.Http())
    service = build('sheets', 'v4', http=http, cache_discovery = False)
    rangeName = 'A1'
    value_input_option = "USER_ENTERED"
    ip = [input["date"],input["time"],input["text"]]
    foos = []
    for foo in input["food"]:
        foos.append(foo)
    ip.append(str(foos).strip('[').strip(']').replace('\'', ''))
    algs = []
    for alg in input["allergens_log"]:
        algs.append(alg)
    ip.append(str(algs).strip('[').strip(']').replace('\'', ''))
    for nutrient in input["nutrients"]:
        ip.append(nutrient)
    payload = {"values": [ip]}
    #print(payload)
    request = service.spreadsheets().values().append(spreadsheetId=spreadsheetId, range=rangeName,
                                                     valueInputOption=value_input_option,body=payload)
    response = request.execute()


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
)


def main():
    assistant = aiy.assistant.grpc.get_assistant()
    with aiy.audio.get_recorder():
         aiy.audio.say('What food did you eat today?', lang="en-US")
         print('Listening...')
         text, audio = assistant.recognize()
         if text:
             cupertino = timezone('US/Pacific')
             now = datetime.now(cupertino)
             date = now.strftime("%B %d, %Y")
             time = now.strftime("%I:%M %p")
             text2 = text.replace("I", "You") + ' on ' + date + ' at ' + time
             aiy.audio.say(text2, lang="en-US")
             tokens = word_tokenize(text)
             tags = pos_tag(tokens)
            # print(tags)
             search_list = []
             url_food = "https://api.nal.usda.gov/ndb/search/"
             for word,tag in tags:
                 if tag == 'NN' or tag == 'NNP' or tag == 'NNS' or tag == 'NNPS':
                     search_list.append(word)
             #print(search_list)
             food_list = []
             if search_list:
                 for word in search_list:
                     querystring = {"format": "json", "q": word, "sort": "n", "max": "25", "offset": "0", "api_key": "nYWMDcdIdc9jiysWJ1V63m2klecwMtcO1PTR7IAh"}
                     response = requests.request("GET", url_food, params=querystring)
                     #print(response.text)
                     resp_text = json.loads(response.text)
                     if "errors" not in resp_text:
                        food_list.append(word)
                        # find the ndb number of the food
                        ndb_number = resp_text["list"]["item"][0]["ndbno"]
                        # print(ndb_number)
                        url_nutrients = "https://api.nal.usda.gov/ndb/reports/"

                        number = str(ndb_number)
                        querystring_nutrients = {"ndbno": number, "type": "b", "format": "json", "api_key": "nYWMDcdIdc9jiysWJ1V63m2klecwMtcO1PTR7IAh"}

                        headers_nutrients = {
                                'cache-control': "no-cache"
                        }

                        response_nutrients = requests.request("GET", url_nutrients, headers=headers_nutrients, params=querystring_nutrients)

                        # get nutrient list of the food
                        response_text_nutrients = json.loads(response_nutrients.text)

                        # get ingredients of the food
                        ingredients_list = response_text_nutrients["report"]["food"]["ing"]["desc"]

                        # find allergens in the food
                        for allergen in allergens:
                           if allergen.upper() in ingredients_list:
                              allergens_list.append(allergen)

                        # print(allergens_list)

                        # get all nutrient values
                        nutrient_list = []
                        for nutrient in logged:
                            value = 0
                            for i in range(0, len(response_text_nutrients["report"]["food"]["nutrients"])):
                                if (nutrient == response_text_nutrients["report"]["food"]["nutrients"][i]["name"]):
                                    value = response_text_nutrients["report"]["food"]["nutrients"][i]["value"]
                            #print(nutrient + ": " + str(value))
                            nutrient_list.append(value)

            #print(food_list)
             credentials = sheet_oauth()
             input = {"date":date,"time":time,"text":text,"food":food_list, "allergens_log": allergens_list, "nutrients": nutrient_list}
             write_to_sheet(credentials,'1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', input)
if __name__ == '__main__':
    main()
