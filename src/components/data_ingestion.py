"""import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('Artifacts','train.csv')
    test_data_path :str = os.path.join('Artifacts','test.csv')
    raw_data_path :str = os.path.join('Artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train_test_split initited')
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=33)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )      
        except  Exception as e:
            raise CustomException(e,sys)
            


''''
if __name__ == '__main__':
    print("✅ Script has started")  # Add this line first
    ingestion = DataIngestion()
    train_path, test_path  = ingestion.initiate_data_ingestion()  # Make sure function name is correct
    print(f"✅ Train Path: {train_path}")  # Confirm train path is returned
    print(f"✅ Test Path: {test_path}")    # Confirm test path is returned


    data_transformation = DataTransformation()
    train_arr, test_arr, processor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    print("✅ Data Transformation Done")
'''
'''
if __name__ == '__main__':
    print("✅ Script has started")
    
    ingestion = DataIngestion()
    train_path, test_path  = ingestion.initiate_data_ingestion()
    print(f"✅ Train Path: {train_path}")
    print(f"✅ Test Path: {test_path}")

    # Trigger transformation
    try:
        print("✅ About to start data transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, processor_path = data_transformation.initiate_data_transformation(train_path, test_path)

        print("✅ Data Transformation Done")
        print(f"Train Array Shape: {train_arr.shape}")
        print(f"Test Array Shape: {test_arr.shape}")
        print(f"Processor saved at: {processor_path}")

    except Exception as e:
        print("❌ Data Transformation Failed")
        print(e)
'''

if __name__ == '__main__':
    print("✅ Script has started")  # Sanity check
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"✅ Train Path: {train_path}")
    print(f"✅ Test Path: {test_path}")

    print("✅ About to start data transformation")  # Debug print
    data_transformation = DataTransformation()
    train_arr, test_arr, processor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    print("✅ Data Transformation Done")

"""

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


from src.components.model_trainer import ModelTrainer




@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('Artifacts', 'train.csv')
    test_data_path: str = os.path.join('Artifacts', 'test.csv')
    raw_data_path: str = os.path.join('Artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('train_test_split initiated')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=33)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

'''
if __name__ == '__main__':
    print("✅ Script has started")
    
    # Data Ingestion Process
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()  
    print(f"✅ Train Path: {train_path}")
    print(f"✅ Test Path: {test_path}")
    
    # Data Transformation Process
    try:
        print("✅ About to start data transformation")
        data_transformation = DataTransformation()
        
        # Start the transformation with the data paths
        train_arr, test_arr, processor_path = data_transformation.initiate_data_transformation(train_path, test_path)

        print("✅ Data Transformation Done")
        print(f"Train Array Shape: {train_arr.shape}")
        print(f"Test Array Shape: {test_arr.shape}")
        print(f"Processor saved at: {processor_path}")
    
    except Exception as e:
        print("❌ Data Transformation Failed")
        print(e)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
'''
'''    
if __name__ == '__main__':
    print("✅ Script has started")
    
    # Data Ingestion Process
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()  
    print(f"✅ Train Path: {train_path}")
    print(f"✅ Test Path: {test_path}")
    
    # Data Transformation and Model Training Process
    try:
        print("✅ About to start data transformation")
        data_transformation = DataTransformation()
        
        # Start the transformation with the data paths
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)

        print("✅ Data Transformation Done")
        print(f"Train Array Shape: {train_arr.shape}")
        print(f"Test Array Shape: {test_arr.shape}")
       #print(f"Processor saved at: {processor_path}")

        # Model Training
        modeltrainer = ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        print("❌ Process Failed")
        print(e) 
''' '''
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array, preprocessor_path))'''

if __name__ == '__main__':
    print("✅ Script has started")

    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"✅ Train Path: {train_path}")
    print(f"✅ Test Path: {test_path}")

    try:
        # Step 2: Data Transformation
        print("✅ About to start data transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, processor_path = data_transformation.initiate_data_transformation(train_path, test_path)
        print("✅ Data Transformation Done")
        print(f"Train Array Shape: {train_arr.shape}")
        print(f"Test Array Shape: {test_arr.shape}")
        print(f"Processor saved at: {processor_path}")

        # Step 3: Model Training
        print("✅ About to start model training")
        model_trainer = ModelTrainer()
        r2_score_result = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"✅ Model Training Done. R2 Score: {r2_score_result}")

    except Exception as e:
        print("❌ Process Failed")
        print(e)

