'''
Raj Palleti
Last revision: 8/15/19

This class gets foods, allergens, and nutrients
from the user's text using the Edamam Food API.

'''

import requests
import json
import sys

class foodProcessor(object):

    def __init__(self,appID = None,appKey = None):
        '''
        Initialize with appID and appKey (needed to make requests to Edamam Food Database)
        '''

        self.appID = appID
        self.appKey = appKey

    def get_food_list(self, rawText = ''):
        '''
        This method returns list of foods, foodURIs, and measureURIs given rawText
        by making a request to the Edamam Food Database and then
        calling parseResponse() to return foods, foodURIs, and measureURIs.
        rawText stores the user's text.
        '''

        text = rawText
        url = "https://api.edamam.com/api/food-database/parser"

        querystring = {"nutrition-type": "logging", "ingr": text, "app_id": "6bb24f34",
                       "app_key": "bcd38e86ec9f271288974f431e0c94e6"}
        response = requests.request("GET", url, params=querystring)

        if response.status_code == 200:
            return self.__parseResponse(response.text)
        else:
            print(response.status_code, response.text)
            sys.exit(0)


    @staticmethod
    def __parseResponse(response):
        '''
        This method returns list of foods, foodURIs, and measureURIs.
        response stores the response text from the request to the Edamam Food Database.
        '''

        foods = []
        foodIds = []
        measureURIs = []
        quantities = []
        response_text = json.loads(response)

        food_list = response_text["parsed"]

        quantity = 0

        if len(food_list) == 0:
            '''
            If no foods have been recognized, use the first item from
            the list of hints, which provides close matches for food items.
            '''
            foods.append(response_text["hints"][0]["food"]["label"])
            foodIds.append(response_text["hints"][0]["food"]["foodId"])
            measures = response_text["hints"][0]["measures"]
            measureURIs.append(measures[int((0 + len(measures) - 1)/2)]["uri"])
        else:
            for food in food_list:

                if "measure" in food:
                    foods.append(food["food"]["label"])
                    foodIds.append(food["food"]["foodId"])
                    measureURIs.append(food["measure"]["uri"])

                    if (food["measure"]["label"] == "Serving"):
                        measureURI = food["measure"]["uri"]
                        foodId = food["food"]["foodId"]

                        url = "https://api.edamam.com/api/food-database/nutrients"
                        querystring = {"app_id": "6bb24f34", "app_key": "bcd38e86ec9f271288974f431e0c94e6"}
                        payload = {"yield": 1,
                                   "ingredients": [{"quantity": 1, "measureURI": measureURI, "foodId": foodId}]}
                        headers = {
                            'content-type': "application/json",
                            'cache-control': "no-cache",
                            'postman-token': "a7ec9f59-5d3e-33d7-e5fb-4358907a8c41"
                        }

                        response = requests.request("POST", url, data=json.dumps(payload), headers=headers,
                                                    params=querystring)

                        if response.status_code == 200:
                            response_text = json.loads(response.text)
                            quantity = str(round(response_text["totalWeight"], 1)) + " Grams"
                            quantities.append(quantity)

                    else:
                        quantity = str(round(food["quantity"], 1)) + " "
                        quantity += food["measure"]["label"] + "s"
                        quantities.append(quantity)

        return foods, foodIds, measureURIs, quantities

    def get_food_details(self,foodIds,measureURIs):
        '''
        This method returns the nutrients and allergens for all foods.
        foodURIs and measureURIs are used to make requests to the Edamam Food Database.
        '''

        details=[]
        for i in range(len(foodIds)):
            details.append(self.get_nutrients_and_allergens(foodIds[i], measureURIs[i]))
        return details


    def get_nutrients_and_allergens(self,foodId,measureURI):
        '''
        This method returns the nutrients and allergens of a single food.
        foodURI and measureURI are used to make requests to the Edamam Food Database to obtain nutrients and allergens.
        '''

        url = "https://api.edamam.com/api/food-database/nutrients"
        querystring = {"app_id": "6bb24f34", "app_key": "bcd38e86ec9f271288974f431e0c94e6"}
        payload = {"yield": 1,
                   "ingredients": [{"quantity": 1, "measureURI": measureURI, "foodId": foodId}]}
        headers = {
            'content-type': "application/json",
            'cache-control': "no-cache",
            'postman-token': "a7ec9f59-5d3e-33d7-e5fb-4358907a8c41"
        }

        response = requests.request("POST", url, data=json.dumps(payload), headers=headers, params=querystring)

        # convert stores the key to translate codes for each nutrient to the nutrients themselves
        convert = {"ENERC_KCAL": "Energy", "FAT": "Fat", "CHOCDF": "Carbs", "FIBTG": "Fiber", "SUGAR": "Sugars", "PROCNT": "Protein", "NA": "Sodium", "CA": "Calcium", "MG": "Magnesium", "K": "Potassium", "FE": "Iron", "VITC": "Vitamin C", "TOCPHA": "Vitamin E", "VITK1": "Vitamin K"}

        # nutrients contains the values for each nutrient
        nutrients = {"Energy": 0, "Fat": 0, "Carbs": 0, "Fiber": 0, "Sugars": 0, "Protein": 0, "Sodium": 0, "Calcium": 0, "Magnesium": 0, "Potassium": 0, "Iron": 0, "Vitamin C": 0, "Vitamin E": 0, "Vitamin K": 0}

        # allergens contains all of the allergens of a given food
        allergens = {}

        if response.status_code == 200:
            response_text = json.loads(response.text)


            if response_text["ingredients"][0]["parsed"][0]["status"] == "MISSING_QUANTITY":
                measureURI = "http://www.edamam.com/ontologies/edamam.owl#Measure_serving"
                payload = {"yield": 1,
                           "ingredients": [{"quantity": 1, "measureURI": measureURI, "foodId": foodId}]}
                response = requests.request("POST", url, data=json.dumps(payload), headers=headers, params=querystring)
                response_text = json.loads(response.text)

            # if the nutrient exists, then record the quantity of nutrient
            for nutrient in response_text["totalNutrients"]:
                if nutrient in convert:
                    nutrients[convert[nutrient]] = round(response_text["totalNutrients"][nutrient]["quantity"], 1)

            allergens = self.get_allergens(response_text)

        else:
            print("Response Status code:", response.status_code, "Response:", response.text)

        return {"nutrients":nutrients,"allergens":allergens}

    def get_allergens(self,response_text):
        '''
        This method returns the allergens of a single food.
        response_text stores the response from the POST request to the Edamam Food Database, from which allergens are extracted.
        '''

        possibleAllergens = ["WHEAT_FREE", "EGG_FREE", "MILK_FREE", "PEANUT_FREE", "TREE_NUT_FREE", "SOY_FREE", "FISH_FREE", "SHELLFISH_FREE"]

        # allergens contains all the allergens of the given food by taking the set difference between all possible allergens and the given health labels
        allergens = set(possibleAllergens).difference(set(response_text["healthLabels"]))
        allergens = list(allergens)

        # format allergens to remove the 'free' label from each allergen
        for i in range(len(allergens)):
            allergens[i] = allergens[i][:-5]
        return allergens
