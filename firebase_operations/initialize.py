import firebase_admin
from firebase_admin import credentials

def initialize_firebase():
    cred = credentials.Certificate("credentials/firebase_credentials.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'nectar-bf0a6.firebasestorage.app'
    })