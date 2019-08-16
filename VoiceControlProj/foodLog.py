'''
Raj Palleti
Last revision: 8/15/19

This class handles Google Sheets authorization
and logging information onto the Google Spreadsheet.

'''

import json
import urllib
import oauth2client
from googleapiclient.discovery import build
import httplib2

class foodLog():
    def __init__(self,key=None):
        self.key =key
        return

    def sheet_oauth(self):
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
                                             'grant_type': 'refresh_token',
                                             'client_id': client_id,
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


    def write_to_sheet(self,credentials, spreadsheetId, sheetName, payload):
        '''
        This method appends the input onto the given spreadsheet.
        '''

        http = credentials.authorize(httplib2.Http())
        service = build('sheets', 'v4', http=http, cache_discovery=False)
        rangeName = sheetName

        value_input_option = "USER_ENTERED"

        request = service.spreadsheets().values().append(spreadsheetId=spreadsheetId, range=rangeName,
                                                         valueInputOption=value_input_option, body=payload)
        response = request.execute()


    def create_new_sheet(self, credentials, spreadsheetId, sheetName):
        '''
        This method creates a new sheet in the spreadsheet.
        The new sheet may be created to log the new user's entries
        or to maintain the user's summary log.
        '''

        http = credentials.authorize(httplib2.Http())
        service = build('sheets', 'v4', http=http, cache_discovery=False)

        batch_update_spreadsheet_request_body = {
            # A list of updates to apply to the spreadsheet.
            # Requests will be applied in the order they are specified.
            # If any request is not valid, no requests will be applied.
            'requests': [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheetName
                        }
                    }
                }
            ],  # TODO: Update placeholder value.

            # TODO: Add desired entries to the request body.
        }

        request = service.spreadsheets().batchUpdate(spreadsheetId=spreadsheetId,
                                                     body=batch_update_spreadsheet_request_body)
        response = request.execute()

        ip = []

        # If the given sheet is used as a summary log, append only the following nutrient totals
        if "_Log" in sheetName:
            ip.append('Summary Date')
            ip.append('Allergens Consumed')
            ip.append('Total Energy (kcal)')
            ip.append('Total Protein (g)')
            ip.append('Total Fat (g)')
            ip.append('Total Sugar (g)')

        # If the given sheet is not used as a summary log,
        # then append all information including date, time, text, foods, nutrients, and allergens into the user's running sheet.
        else:
            ip.append('Date')
            ip.append('Time')
            ip.append('Text')
            ip.append('Food')
            ip.append('Quantity')
            ip.append('Allergens')
            ip.append('Energy (kcal)')
            ip.append('Fat (g)')
            ip.append('Carbs (g)')
            ip.append('Fiber (g)')
            ip.append('Sugars (g)')
            ip.append('Protein (g)')
            ip.append('Sodium (mg)')
            ip.append('Calcium (mg)')
            ip.append('Magnesium (mg)')
            ip.append('Potassium (mg)')
            ip.append('Iron (mg)')
            ip.append('Vitamin C (mg)')
            ip.append('Vitamin E (mg)')
            ip.append('Vitamin K (µg)')

        payload = {"values": [ip]}
        self.write_to_sheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', sheetName, payload)


    def isInSheet(self, credentials, spreadsheetId, name):
        '''
        Determine if a sheet with a given name exists in the spreadsheet.
        '''

        http = credentials.authorize(httplib2.Http())
        service = build('sheets', 'v4', http=http, cache_discovery=False)

        allSheets = service.spreadsheets().get(spreadsheetId=spreadsheetId).execute().get('sheets', )

        for sheet in allSheets:
            # get the title of each sheet
            sheetTitle = sheet.get("properties", {}).get("title")
            if sheetTitle == name:
                return True

        return False


    def readSheet(self, credentials, spreadsheetId, sheetName, range):
        '''
        Method to read and return values in a given range from a given sheet of the spreadsheet.
        '''

        http = credentials.authorize(httplib2.Http())
        service = build('sheets', 'v4', http=http, cache_discovery=False)
        rangeName = sheetName

        value_input_option = "USER_ENTERED"

        request = service.spreadsheets().values().get(spreadsheetId=spreadsheetId, range= sheetName+"!"+range)
        response = request.execute()
        values = response.get('values', [])
        return values


    def process_values(self, credentials, values, date, numFoods, sheetName):
        '''
        Read values from a user's sheet and return daily totals to be logged onto the user's summary log sheet.
        '''

        indexes = []
        for i, j in enumerate(values):
            for d in j:
                if d == date:
                    indexes.append(i + 1)

        algsTotal = []
        engTotal = 0
        protTotal = 0
        fatTotal = 0
        sugarsTotal = 0

        algsData = []
        engData = []
        if len(indexes):
            temp = "F" + str(indexes[0])+":F" + str(indexes[len(indexes) - 1] + numFoods)
            allData = self.readSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', sheetName, temp)
            for j in allData:
                if len(j):
                    if (j[0] != "NONE" and j[0] != ''):
                        algsData.append(j[0])

            if not len(algsData):
                algsData.append("NONE")
            algsTotal = '\n'.join(set(algsData))

            temp = "G" + str(indexes[0])+":G" + str(indexes[len(indexes) - 1] + numFoods)
            allData = self.readSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', sheetName, temp)
            for j in allData:
                if len(j):
                    if "Total" in j[0]:
                        total = j[0].split(": ")
                        if len(total) > 1:
                            engTotal += round(float(total[1]), 1)

            temp = "L" + str(indexes[0]) + ":L" + str(indexes[len(indexes) - 1] + numFoods)
            allData = self.readSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', sheetName, temp)
            for j in allData:
                if len(j):
                    if "Total" in j[0]:
                        total = j[0].split(": ")
                        if len(total) > 1:
                            protTotal += round(float(total[1]), 1)

            temp = "H" + str(indexes[0]) + ":H" + str(indexes[len(indexes) - 1] + numFoods)
            allData = self.readSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', sheetName, temp)
            for j in allData:
                if len(j):
                    if "Total" in j[0]:
                        total = j[0].split(": ")
                        if len(total) > 1:
                            fatTotal += round(float(total[1]), 1)

            temp = "K" + str(indexes[0]) + ":K" + str(indexes[len(indexes) - 1] + numFoods)
            allData = self.readSheet(credentials, '1GxFpWhwISzni7DWviFzH500k9eFONpSGQ8uJ0-kBKY4', sheetName, temp)
            for j in allData:
                if len(j):
                    if "Total" in j[0]:
                        total = j[0].split(": ")
                        if len(total) > 1:
                            sugarsTotal += round(float(total[1]), 1)

        ip2 = []
        ip2.append(date)
        ip2.append(algsTotal)
        ip2.append(engTotal)
        ip2.append(protTotal)
        ip2.append(fatTotal)
        ip2.append(sugarsTotal)

        payload = {"values": [ip2]}
        return payload


    def update_sheet(self, credentials, spreadsheetId, sheetName, index_date, payload, dateExists):
        '''
        Method to update a user's summary log sheet with the user's total nutrients consumed for the given day.
        '''

        http = credentials.authorize(httplib2.Http())
        service = build('sheets', 'v4', http=http, cache_discovery=False)
        rangeName = sheetName

        value_input_option = "USER_ENTERED"

        if dateExists == False:
            rangeName = sheetName
            request = service.spreadsheets().values().append(spreadsheetId=spreadsheetId, range=rangeName,
                                                             valueInputOption=value_input_option, body=payload)
        else:
            temp = sheetName + "!A" + str(index_date) + ":F" + str(index_date)
            request = service.spreadsheets().values().update(spreadsheetId=spreadsheetId, range=temp,
                                                         valueInputOption=value_input_option, body=payload)

        response = request.execute()
        
