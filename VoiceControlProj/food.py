'''
Raj Palleti
Last revision: 2/22/19

This class instantiates foodProcessor to get foods, allergens, and nutrients 
from the user's text, which is recognized using the Google Voice Kit.
It then uses foodLog to log the information onto a Google Spreadsheet.

'''

import foodProcessor
import foodLog
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
)


def main():

    '''
    "id" and "key" are used to make requests to the Edamam Food API
    and they are obtained by registering for an account from Edamam.
    '''
    id = '5ce56395'
    key = 'da9676a9e9fefcbb46be59b59f20bf80'

    assistant = aiy.assistant.grpc.get_assistant()
    with aiy.audio.get_recorder():
         aiy.audio.say('What food did you eat today?', lang="en-US")
         print('Listening...')
         text, audio = assistant.recognize()
         if text:
             # find date and time
             cupertino = timezone('US/Pacific')
             now = datetime.now(cupertino)
             date = now.strftime("%B %d, %Y")
             time = now.strftime("%I:%M %p")
             text2 = text.replace("I", "You") + ' on ' + date + ' at ' + time
             aiy.audio.say(text2, lang="en-US")

             rawText = text

             # Instantiate foodProcessor
             processor = foodProcessor.foodProcessor(key,id)

             '''
             Get list of foods, foodURIs, and measureURIs for each food.
             foodURIs and measureURIs are used to get the nutrients of each food.
             '''
             foods, foodURIs, measureURIs = processor.getFoodList(rawText)

             # Get allergens and nutrients from all foods
             details = processor.getFoodDetails(foodURIs, measureURIs)

             # Instantiate foodLog
             flog = foodLog.foodLog()

             allergens = []

             nutrientsToLog = ["Energy", "Fat", "Carbs", "Fiber", "Sugars", "Protein", "Sodium", "Calcium", "Magnesium", "Potassium", "Iron", "Vitamin C", "Vitamin E", "Vitamin K"]
             # nutrients contains the values for each nutrient
             nutrients = {"Energy": [], "Fat": [], "Carbs": [], "Fiber": [], "Sugars": [], "Protein": [], "Sodium": [], "Calcium": [], "Magnesium": [], "Potassium": [], "Iron": [], "Vitamin C": [], "Vitamin E": [], "Vitamin K": []}

             for food in details:
                 allergens.append(food["allergens"])
                 for nutrient in nutrientsToLog:
                    nutrients[nutrient].append(food["nutrients"][nutrient])

             credentials = flog.sheet_oauth()

             # ip contains the values that will be appended onto the next row of the Google Spreadsheet
             ip = []
             ip.append(date)
             ip.append(time)
             ip.append(rawText)

             if len(foods) > 0:
                '''
                Log the date, time, text, nutrient and allergen info of
                the first food onto the Google Spreadsheet.
                '''
                ip.append(foods[0])
                if len(allergens[0]) == 0:
                   ip.append("NONE")
                else:
                   ip.append(', '.join(allergens[0]))
                for nutrient in nutrientsToLog:
                   ip.append(nutrients[nutrient][0])
                payload = {"values": [ip]}
                flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', payload)

                '''
                Log the nutrient and allergen info of each subsequent
                food onto a new row of the Google Spreadsheet.
                '''
                for i in range(1,len(foods)):
                   ip = ["", "", "", foods[i]]
                   if len(allergens[i]) == 0:
                      ip.append("NONE")
                   else:
                      ip.append(', '.join(allergens[i]))
                   for nutrient in nutrientsToLog:
                      ip.append(nutrients[nutrient][i])
                   payload = {"values": [ip]}
                   flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', payload)

                # Log the nutrient totals onto a new row of the Google Spreadsheet.
                ip = ["", "", "", "", ""]
                for nutrient in nutrientsToLog:
                   total = 0
                   for quantity in nutrients[nutrient]:
                      total += quantity
                   ip.append("Total: " + str(round(total,1)))
                payload = {"values": [ip]}
                flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', payload)
             else:
                # no foods have been recognized
                ip.append("NONE")
                ip.append("NONE")
                for nutrient in nutrientsToLog:
                   ip.append("Total: 0")
                payload = {"values": [ip]}
                flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', payload)

if __name__ == '__main__':
    main()
