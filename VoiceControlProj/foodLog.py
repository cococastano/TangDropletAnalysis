'''
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

    def write_to_sheet(self,credentials, spreadsheetId, payload):
        '''
        This method appends the input onto the given spreadsheet.
        '''
        http = credentials.authorize(httplib2.Http())
        service = build('sheets', 'v4', http=http, cache_discovery=False)
        rangeName = 'A1'
        value_input_option = "USER_ENTERED"
        request = service.spreadsheets().values().append(spreadsheetId=spreadsheetId, range=rangeName,
                                                         valueInputOption=value_input_option, body=payload)
        response = request.execute()

