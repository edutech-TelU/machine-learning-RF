import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('edutech-dev-app-firebase-adminsdk-cwnu3-29a6346ee9.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
