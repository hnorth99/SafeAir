import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
import pytz

location_id = "HunterApt"


firebase_path = "safeairdevice-firebase-adminsdk-ie22z-1ae09cca74.json"
cred = credentials.Certificate(firebase_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://safeairdevice-default-rtdb.firebaseio.com/'
})

def new_location(location):
    firebase_ref = db.reference('/' + location)
    firebase_ref.set({
        "currentcapacity": 0,
        "maxcapacity": 20,
        "capacitycomplaints": 0,
        "co2": None,
        "co2complaints": 0,
        "humidity": None,
        "humiditycomplaints": 0,
        "temperature": None,
        "temperaturecomplaints": 0,
        "time": None,
        "password": "abc123",
        "size": 0
        
    })

def read_location(location, metric = ""):
    firebase_ref = db.reference('/' + location + '/' + metric)
    return firebase_ref.get()
    
def climate_update(location, currentcapacity, co2, humidity, temperature):
    firebase_ref = db.reference('/' + location)
    size = read_location(location, "size")
    tz_pst = pytz.timezone('US/Pacific')
    time = datetime.now(tz_pst).strftime("%H:%M:%S")
    
    firebase_ref.update({
        "currentcapacity": currentcapacity,
        "co2/" + str(size): co2,
        "humidity/" + str(size): humidity,
        "temperature/" + str(size): temperature,
        "time/" + str(size): time,
        "size": size + 1
    })
        
new_location(location_id)