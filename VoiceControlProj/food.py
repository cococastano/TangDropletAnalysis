'''
Raj Palleti
Last revision: 8/16/19

This class instantiates foodProcessor to get each person's foods, allergens, 
and nutrients from the user's text, which is recognized using the Google Voice Kit.
It then uses foodLog to log each person's dietary info into their own sheet in a Google Spreadsheet.

'''

# Set the 'JAVAHOME' variable to your own path of jdk/bin/java. 
import os
os.environ['JAVAHOME'] = "/usr/java/jdk1.8.0_202/bin/java"

import aiy.assistant.grpc
import aiy.audio
import aiy.voicehat
import sys

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import gender_guesser.detector as gender

import foodProcessor
import foodLog
from datetime import datetime
from pytz import timezone


def convert_pronouns_to_names(text, username):
    '''
    This method uses Stanford NER and the gender-guesser package to convert pronouns in the text to 
    their corresponding names. It returns the processed text after replacing all pronouns.
    '''

    # Set your own path for the classification model and Stanford tagged jar file of StanfordNERTagger. 
    st = StanfordNERTagger(
        '/home/pi/AIY-voice-kit-python/src/examples/voice/app/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
        '/home/pi/AIY-voice-kit-python/src/examples/voice/app/stanford-ner-2018-10-16/stanford-ner.jar',
        encoding='utf-8')

    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)

    det = gender.Detector()

    wordCount = len(tokenized_text)

    # Keep track of the most recent male and female names, which will be used to replace the pronouns "He" and "She".
    lastMaleName = ''
    lastFemaleName = ''
    index = 0
    newText = ''

    for i in range(wordCount):

        word = classified_text[i][0]
        partOfSpeech = classified_text[i][1]

        if word == 'He' or word == 'he':

            if lastMaleName != '':
                newText = newText + lastMaleName

            else:
                newText = newText + word

        elif word == 'She' or word == 'she':

            if lastFemaleName != '':
                newText = newText + lastFemaleName

            else:
                newText = newText + word

        elif word == 'I':

            newText = newText + username

        else:

            newText = newText + word

        if partOfSpeech == 'PERSON':

            if "female" in det.get_gender(word):

                lastFemaleName = word

            elif "male" in det.get_gender(word):

                lastMaleName = word

        index = index + len(word)

        if index < len(text) and text[index] == ' ':
            index = index + 1
            newText += ' '

    return newText


def get_substrings(text):
    '''
    This method splits the text into substrings, where each begins with a name 
    and continues until reaching the next name. It will return the list of substrings 
    and a list that contains the name in each substring. 
    '''

    # Set your own path for the classification model and Stanford tagged jar file of StanfordNERTagger. 
    st = StanfordNERTagger(
        '/home/pi/AIY-voice-kit-python/src/examples/voice/app/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
        '/home/pi/AIY-voice-kit-python/src/examples/voice/app/stanford-ner-2018-10-16/stanford-ner.jar',
        encoding='utf-8')

    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)

    wordCount = len(tokenized_text)

    # charIndexes stores the starting indexes for each name from the text. 
    charIndexes = []
    charCounter = 0
    newCharCounter = 0
    substrings = []
    names = []

    for i in range(wordCount):

        word = classified_text[i][0]
        partOfSpeech = classified_text[i][1]

        if partOfSpeech == 'PERSON':
            newCharCounter = text.find(word, charCounter)
            charIndexes.append(newCharCounter)
            charCounter = newCharCounter + 1
            names.append(classified_text[i][0])

    for i in range(len(charIndexes)):
        currIndex = charIndexes[i]
        if i == len(charIndexes) - 1:
            substrings.append(text[currIndex: ])
        else:
            nextIndex = charIndexes[i + 1]
            substrings.append(text[currIndex: nextIndex])

    return substrings, names


def get_diet(substrings, names):
    '''
    This method uses the substrings to determine the foods, nutrients, and allergens consumed by each person.
    It will return a dictionary containing the dietary information for each person.
    '''

    '''
    "id" and "key" are used to make requests to the Edamam Food API,
    and they are obtained by registering for an account from Edamam.
    '''
    id = '6bb24f34'
    key = 'bcd38e86ec9f271288974f431e0c94e6'

    diet = {}
    for name in names:
        if name not in diet:
            diet[name] = {}
            diet[name]['foods'] = []
            diet[name]['quantities'] = []
            diet[name]['allergens'] = []
            diet[name]['nutrients'] = {"Energy": [], "Fat": [], "Carbs": [], "Fiber": [], "Sugars": [], "Protein": [], "Sodium": [], "Calcium": [],
                                       "Magnesium": [], "Potassium": [], "Iron": [], "Vitamin C": [], "Vitamin E": [], "Vitamin K": []}

    # For each substring, find the person's name and update the person's dietary information using the foods in the substring. 
    for i in range(len(substrings)):

        substring = substrings[i]
        name = names[i]

        # Instantiate foodProcessor.
        processor = foodProcessor.foodProcessor(key, id)
        
        '''
        Get list of foods, foodURIs, measureURIs, and quantities for each food.
        foodURIs and measureURIs are used to get the nutrients for each food.
        '''
        foods, foodIds, measureURIs, quantities = processor.get_food_list(substring)

        # Get allergens and nutrients from all foods.
        details = processor.get_food_details(foodIds, measureURIs)

        allergens = []
        nutrients = {"Energy": [], "Fat": [], "Carbs": [], "Fiber": [], "Sugars": [], "Protein": [], "Sodium": [],
                     "Calcium": [], "Magnesium": [], "Potassium": [], "Iron": [], "Vitamin C": [], "Vitamin E": [],
                     "Vitamin K": []}

        # Add the foods and quantities to the person's diet. 
        diet[name]['foods'].extend(foods)
        diet[name]['quantities'].extend(quantities)

        # For each food, add the allergens and nutrients to the person's diet. 
        for i in range(len(details)):

            food = details[i]
            diet[name]['allergens'].append(format_allergens(food['allergens']))

            for nutrient in nutrients:
                diet[name]['nutrients'][nutrient].append(food["nutrients"][nutrient])

    return diet


def format_allergens(allergens):
    '''
    This method concatenates the list of allergens in each food to a string.
    '''

    if len(allergens) == 1:
        return allergens[0]

    algs = ''
    for i in range(len(allergens)):

        for j in range(len(allergens[i])):
            if j == len(allergens[i]) - 1:
                algs += allergens[i][j]
                if i != len(allergens) - 1:
                    algs += ', '
            else:
                algs += allergens[i][j]

    return algs


def log_diet(diet, rawText):
    '''
    This method uses the diet dictionary to log the dietary information for each person in the corresponding sheet.
    It will also update everyone's summary log sheet.
    '''

    # Instantiate foodLog
    flog = foodLog.foodLog()
    cupertino = timezone('US/Pacific')
    now = datetime.now(cupertino)
    date = now.strftime("%B %d, %Y")
    time = now.strftime("%I:%M %p")

    credentials = flog.sheet_oauth()

    for name in diet:

        # ip contains the values that will be appended onto the next row of the Google Spreadsheet.
        ip = []
        ip.append(date)
        ip.append(time)
        ip.append(rawText)

        '''
        If the person consumed at least one food item, then construct a new row
        containing dietary information to be logged in the person's sheet. 
        '''
        if len(diet[name]['foods']) > 0:

            ip.append(diet[name]['foods'][0])
            ip.append(diet[name]['quantities'][0])

            if len(diet[name]['allergens'][0]) == 0:
                ip.append("NONE")
            else:
                ip.append(diet[name]['allergens'][0])

            for nutrient in diet[name]['nutrients']:
                ip.append(diet[name]['nutrients'][nutrient][0])

            payload = {"values": [ip]}
            flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', name, payload)

            for i in range(1, len(diet[name]['foods'])):

                ip = ["", "", "", diet[name]['foods'][i], diet[name]['quantities'][i]]

                if len(diet[name]['allergens'][i]) == 0:
                    ip.append("NONE")
                else:
                    ip.append(diet[name]['allergens'][i])

                for nutrient in diet[name]['nutrients']:
                    ip.append(diet[name]['nutrients'][nutrient][i])

                payload = {"values": [ip]}
                flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', name, payload)

            # Construct a new row containing nutrient totals to be logged in the person's sheet. 
            ip = ["", "", "", "", "", ""]
            for nutrient in diet[name]['nutrients']:
                total = 0
                for quantity in diet[name]['nutrients'][nutrient]:
                    total += quantity
                ip.append("Total: " + str(round(total, 1)))

            payload = {"values": [ip]}
            flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', name, payload)

        # If the person did not consume any foods, then set the nutrient totals to 0 and update the person's sheet. 
        else:
            ip.append("NONE")
            ip.append("NONE")
            ip.append("NONE")
            for nutrient in diet[name]['nutrients']:
                ip.append("Total: 0")
            payload = {"values": [ip]}
            flog.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', name, payload)

        # Read the nutrient values from the person's sheet and update the person's summary log. 
        values = flog.readSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', name, "A1:A10000")
        payload = flog.process_values(credentials, values, date, len(diet[name]['foods']), name)

        daily_log_name = name + "_Daily_Log"
        values = flog.readSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', daily_log_name, "A1:A10000")

        index_date = 1
        dateExists = False
        for i, j in enumerate(values):
            for d in j:
                if d == date:
                    index_date = i + 1
                    dateExists = True

        flog.update_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', daily_log_name, index_date, payload, dateExists)


def main():
   '''
   Prompt the user to enter their name. 
   Create a new sheet for the user if their sheet
   does not already exist in the spreadsheet. 
   Then log the user's diet in their sheet and
   update the user's summary log. 
   '''

   username = input("Please enter your name: ")
   assistant = aiy.assistant.grpc.get_assistant()
   with aiy.audio.get_recorder():
         aiy.audio.say('What food did you or your family members eat today?', lang="en-US")
         print('Listening...')
         text, audio = assistant.recognize()
         if text:
             # find date and time
             cupertino = timezone('US/Pacific')
             now = datetime.now(cupertino)
             date = now.strftime("%B %d, %Y")
             time = now.strftime("%I:%M %p")
             print(text)
             textToUser = text.replace("I", "You")
             textToUser = textToUser.replace("me", "you")
             textToUser = textToUser.replace("my", "your")
             textToUser = textToUser.replace("My", "your")
             textToUser = textToUser.replace("mine", "yours")
             textToUser = textToUser.replace("Mine", "Yours")

             print(textToUser)
             aiy.audio.say(textToUser, lang="en-US")
             flog = foodLog.foodLog()
             credentials = flog.sheet_oauth()

             newText = convert_pronouns_to_names(text, username)

             substrings, names = get_substrings(newText)

             for name in names:
                # Create a new sheet for each person who does not already have a sheet in the Google Spreadsheet. 
                if not flog.isInSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', name):
                   flog.create_new_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', name)
                   daily_log_name = name + "_Daily_Log"
                   flog.create_new_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', daily_log_name)

             diet = get_diet(substrings, names)

             log_diet(diet, text)


if __name__ == '__main__':

    main()
