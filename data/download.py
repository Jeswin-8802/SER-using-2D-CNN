# Note: This file must be run inside the data/ dir

import requests
import os

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/u/0/uc?id=1CjdErMEQ_aITEPOWqTO9o0S_cgtAg7sp&export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = '1CjdErMEQ_aITEPOWqTO9o0S_cgtAg7sp'
    destination = os.path.join(os.getcwd(), 'audio_files.zip')
    download_file_from_google_drive(file_id, destination)
    print('file successfully downloaded into', destination)
