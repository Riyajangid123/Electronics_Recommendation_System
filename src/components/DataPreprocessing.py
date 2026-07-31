import pandas as pd
import numpy as np
import re
    
class DataPreprocessing:
    
    def __init__(self,data):
        self.data=data
        
    def clean_price(self,price):
        return price.astype(str).str.replace("₹","",regex=False).str.replace(",","",regex=False).astype(float)
    
    def clean_text(self,text):
        text = text.lower()

        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        return text
    def preprocess_data(self):
        data=self.data.copy()
        data = data[
            [
                'name',
                'main_category',
                'sub_category',
                'image',
                'link',
                'ratings',
                'no_of_ratings',
                'discount_price',
                'actual_price'
            ]
        ]
        data['discount_price']=self.clean_price(data['discount_price'])
        data['actual_price']=self.clean_price(data['actual_price'])

        data.dropna(subset=["ratings"],inplace=True)
        data["discount_price"]=data["discount_price"].fillna(data["actual_price"])
        
        data['tags'] = (data['name'].astype(str) + " " +
                        data['main_category'].astype(str) + " " +
                        data['sub_category'].astype(str))
        data['tags']=data['tags'].apply(self.clean_text)

        return data





    