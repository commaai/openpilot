import numpy as np
import os
import csv
path = os.path.dirname(os.path.abspath(__file__))
csv_file_name = path + '/rg_cities1000.csv'

# right hand drive is when the steering wheel is on the right of the car
# left hand traffic is when cars driver on the left side of the road
LHT_COUNTRIES = ['AU', 'IN', 'IE', 'JP', 'MU', 'MY', 'NZ', 'UK', 'ZA']


def get_city(lat, lon):
  cities = np.array(list(csv.reader(open(csv_file_name))))[1:]
  positions = cities[:,:2].astype(np.float32)
  idx = np.argmin(np.linalg.norm((positions - np.array([lat, lon])), axis=1))
  return cities[idx]


def is_lht(lat, lon):
  city = get_city(lat, lon)
  country = city[-1]
  return country in LHT_COUNTRIES
