'''
This class gets foods, allergens and nutrients 
from the user's text using the Edamam Food API. 
'''

import requests
import json

class foodProcessor(object):

    def __init__(self,appID = None,appKey = None):
        '''Initialize with app ID and app Key (needed to make requests to Edamam Food Database).'''
        self.appID = appID
        self.appKey = appKey

    def getFoodList(self, rawText = ''):
        '''
        This method returns list of foods, foodURIs, and measureURIs given rawText
        by making a request to the Edamam Food Database and then
        calling parseResponse() to return foods, foodURIs, and measureURIs.
        rawText stores the user's text.
        '''
        text = rawText
        url = "https://api.edamam.com/api/food-database/parser"

        querystring = {"nutrition-type": "logging", "ingr": text, "app_id": "5ce56395",
                       "app_key": "da9676a9e9fefcbb46be59b59f20bf80"}
        response = requests.request("GET", url, params=querystring)
        return self.__parseResponse(response.text)

    @staticmethod
    def __parseResponse(response):
        '''
        This method returns list of foods, foodURIs, and measureURIs.
        response stores the response text from the request to the Edamam Food Database.
        '''
        foods = []
        foodURIs = []
        measureURIs = []
        response_text = json.loads(response)

        food_list = response_text["parsed"]
        for food in food_list:
            foods.append(food["food"]["label"])
            foodURIs.append(food["food"]["uri"])
            measureURIs.append(food["measure"]["uri"])
        return foods, foodURIs, measureURIs

    def getFoodDetails(self,foodURIs,measureURIs):
        '''
        This method returns the nutrients and allergens for all foods.
        foodURIs and measureURIs are used to make requests to the Edamam Food Database.
        '''
        details=[]
        for i in range(len(foodURIs)):
            details.append(self.getInfo(foodURIs[i], measureURIs[i]))
        return details

    def getInfo(self,foodURI,measrueURI):
        '''
        This method returns the nutrients and allergens of a single food.
        foodURI and measureURI are used to make requests to the Edamam Food Database to obtain nutrients and allergens.
        '''
        url = "https://api.edamam.com/api/food-database/nutrients"
        querystring = {"app_id": "5ce56395", "app_key": "da9676a9e9fefcbb46be59b59f20bf80"}
        payload = {"yield": 1,
                   "ingredients": [{"quantity": 1, "measureURI": measrueURI, "foodURI": foodURI}]}
        headers = {
            'content-type': "application/json",
            'cache-control': "no-cache",
            'postman-token': "a7ec9f59-5d3e-33d7-e5fb-4358907a8c41"
        }

        response = requests.request("POST", url, data=json.dumps(payload), headers=headers, params=querystring)

        convert = {"ENERC_KCAL": "Energy", "FAT": "Fat", "CHOCDF": "Carbs", "FIBTG": "Fiber", "SUGAR": "Sugars", "PROCNT": "Protein", "NA": "Sodium", "CA": "Calcium", "MG": "Magnesium", "K": "Potassium", "FE": "Iron", "VITC": "Vitamin C", "TOCPHA": "Vitamin E", "VITK1": "Vitamin K"}

        nutrients = {"Energy": 0, "Fat": 0, "Carbs": 0, "Fiber": 0, "Sugars": 0, "Protein": 0, "Sodium": 0, "Calcium": 0, "Magnesium": 0, "Potassium": 0, "Iron": 0, "Vitamin C": 0, "Vitamin E": 0, "Vitamin K": 0}
        allergens = {}

        if response.status_code == 200:
            response_text = json.loads(response.text)

            #if key exists record quantity of nutrient

            for nutrient in response_text["totalNutrients"]:
                if nutrient in convert:
                    nutrients[convert[nutrient]] = response_text["totalNutrients"][nutrient]["quantity"]

            allergens = self.getAllergens(response_text)
            #print(nutrients)

        else:
            print("Response Status code:", response.status_code, "Response:", response.text)

        return {"nutrients":nutrients,"allergens":allergens}

    def getAllergens(self,response_text):
        '''
        This method returns the allergens of a single food.
        response_text stores the response from the POST request to the Edamam Food Database, from which allergens are extracted.
        '''
        possibleAllergens = ["WHEAT_FREE", "EGG_FREE", "MILK_FREE", "PEANUT_FREE", "TREE_NUT_FREE", "SOY_FREE", "FISH_FREE", "SHELLFISH_FREE"]
        
        # allergens contains all the allergens of the given food by taking the set difference between all possible allergens and the given health labels
        allergens = set(possibleAllergens).difference(set(response_text["healthLabels"]))
        allergens = list(allergens)
        for i in range(len(allergens)):
            allergens[i] = allergens[i][:-5]

        return allergens
