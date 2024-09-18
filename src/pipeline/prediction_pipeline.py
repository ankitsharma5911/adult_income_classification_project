import sys,os,pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            
            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
   

class CustomData:
    def __init__(self,
                 age:float,
                 workclass:str,
                 education:str,
                 marital_status:str,
                 occupation:str,
                 relationship:str,
                 race:str,
                 gender:str,
                 capital_gain:float,
                 capital_loss:float,
                 hours_per_week:float,
                 ):
        self.age = age
        self.workclass = workclass
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.gender = gender
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'capital-gain':[self.capital_gain],
                'capital-loss':[self.capital_loss],
                'hours-per-week':[self.hours_per_week],
                'workclass':[self.workclass],
                'education':[self.education],
                'marital-status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'race':[self.race],
                'gender':[self.gender]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            logging.info(df.head(5))
            return df
        except Exception as e:
            CustomException(e,sys)

 
