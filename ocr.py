import os, io
from google.cloud import vision
from google.oauth2 import service_account
# from google.cloud.vision import types
import pandas as pd

# os.environ['GOOGLE_APP_CREDENTIALS'] = r'ServicAccount_token'
# creds = service_account.Credentials.from_service_account_file('***Use your Google Vision API Key Here****')
creds = service_account.Credentials.from_service_account_file('./ServicAccount_token.json')

# detect the croped image containing the text from the predicted bounding box
# the takes as a parameter the parth to the file from your local disk
# calls the Google Vison API using your personal API key
# passes the image to the Google Vision API and extract and return the text
def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient(credentials=creds,)


    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # extracts the two fields in the json of the API in a pandas DataFrame and 
    # extract only the field containing the text which is the 'description'
    df = pd.DataFrame(columns = ['locale','description'])
    for text in texts:
        df = pd.concat([df, pd.DataFrame({
            'locale': [text.locale],
            'description': [text.description]
        })], ignore_index=True)
    return df['description'][0]