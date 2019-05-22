#!/usr/bin/env python
import json

DEFAULT_OUTPUT_FILENAME = "default_speeds_by_region.json"

def main(filename = DEFAULT_OUTPUT_FILENAME):
  countries = []

  """
  --------------------------------------------------
      US - United State of America
  --------------------------------------------------
  """
  US = Country("US") # First step, create the country using the ISO 3166 two letter code
  countries.append(US) # Second step, add the country to countries list

  """ Default rules """
  # Third step, add some default rules for the country
  # Speed limit rules are based on OpenStreetMaps (OSM) tags.
  # The dictionary {...} defines the tag_name: value
  # if a road in OSM has a tag with the name tag_name and this value, the speed limit listed below will be applied.
  # The text at the end is the speed limit (use no unit for km/h)
  # Rules apply in the order in which they are written for each country
  # Rules for specific regions (states) take priority over country rules
  # If you modify existing country rules, you must update all existing states without that rule to use the old rule
  US.add_rule({"highway": "motorway"}, "65 mph") # On US roads with the tag highway and value motorway, the speed limit will default to  65 mph
  US.add_rule({"highway": "trunk"}, "55 mph")
  US.add_rule({"highway": "primary"}, "55 mph")
  US.add_rule({"highway": "secondary"}, "45 mph")
  US.add_rule({"highway": "tertiary"}, "35 mph")
  US.add_rule({"highway": "unclassified"}, "55 mph")
  US.add_rule({"highway": "residential"}, "25 mph")
  US.add_rule({"highway": "service"}, "25 mph")
  US.add_rule({"highway": "motorway_link"}, "55 mph")
  US.add_rule({"highway": "trunk_link"}, "55 mph")
  US.add_rule({"highway": "primary_link"}, "55 mph")
  US.add_rule({"highway": "secondary_link"}, "45 mph")
  US.add_rule({"highway": "tertiary_link"}, "35 mph")
  US.add_rule({"highway": "living_street"}, "15 mph")

  """ States """
  new_york = US.add_region("New York") # Fourth step, add a state/region to country
  new_york.add_rule({"highway": "primary"}, "45 mph") # Fifth step , add rules to the state. See the text above for how to write rules
  new_york.add_rule({"highway": "secondary"}, "55 mph")
  new_york.add_rule({"highway": "tertiary"}, "55 mph")
  new_york.add_rule({"highway": "residential"}, "30 mph")
  new_york.add_rule({"highway": "primary_link"}, "45 mph")
  new_york.add_rule({"highway": "secondary_link"}, "55 mph")
  new_york.add_rule({"highway": "tertiary_link"}, "55 mph")
  # All if not written by the state, the rules will default to the country rules

  #california = US.add_region("California")
  # California uses only the default US rules

  michigan = US.add_region("Michigan")
  michigan.add_rule({"highway": "motorway"}, "70 mph")

  oregon = US.add_region("Oregon")
  oregon.add_rule({"highway": "motorway"}, "55 mph")
  oregon.add_rule({"highway": "secondary"}, "35 mph")
  oregon.add_rule({"highway": "tertiary"}, "30 mph")
  oregon.add_rule({"highway": "service"}, "15 mph")
  oregon.add_rule({"highway": "secondary_link"}, "35 mph")
  oregon.add_rule({"highway": "tertiary_link"}, "30 mph")

  south_dakota = US.add_region("South Dakota")
  south_dakota.add_rule({"highway": "motorway"}, "80 mph")
  south_dakota.add_rule({"highway": "trunk"}, "70 mph")
  south_dakota.add_rule({"highway": "primary"}, "65 mph")
  south_dakota.add_rule({"highway": "trunk_link"}, "70 mph")
  south_dakota.add_rule({"highway": "primary_link"}, "65 mph")

  wisconsin = US.add_region("Wisconsin")
  wisconsin.add_rule({"highway": "trunk"}, "65 mph")
  wisconsin.add_rule({"highway": "tertiary"}, "45 mph")
  wisconsin.add_rule({"highway": "unclassified"}, "35 mph")
  wisconsin.add_rule({"highway": "trunk_link"}, "65 mph")
  wisconsin.add_rule({"highway": "tertiary_link"}, "45 mph")

  """
  --------------------------------------------------
      AU - Australia
  --------------------------------------------------
  """
  AU = Country("AU")
  countries.append(AU)

  """ Default rules """
  AU.add_rule({"highway": "motorway"}, "100")
  AU.add_rule({"highway": "trunk"}, "80")
  AU.add_rule({"highway": "primary"}, "80")
  AU.add_rule({"highway": "secondary"}, "50")
  AU.add_rule({"highway": "tertiary"}, "50")
  AU.add_rule({"highway": "unclassified"}, "80")
  AU.add_rule({"highway": "residential"}, "50")
  AU.add_rule({"highway": "service"}, "40")
  AU.add_rule({"highway": "motorway_link"}, "90")
  AU.add_rule({"highway": "trunk_link"}, "80")
  AU.add_rule({"highway": "primary_link"}, "80")
  AU.add_rule({"highway": "secondary_link"}, "50")
  AU.add_rule({"highway": "tertiary_link"}, "50")
  AU.add_rule({"highway": "living_street"}, "30")

  """
  --------------------------------------------------
      CA - Canada
  --------------------------------------------------
  """
  CA = Country("CA")
  countries.append(CA)

  """ Default rules """
  CA.add_rule({"highway": "motorway"}, "100")
  CA.add_rule({"highway": "trunk"}, "80")
  CA.add_rule({"highway": "primary"}, "80")
  CA.add_rule({"highway": "secondary"}, "50")
  CA.add_rule({"highway": "tertiary"}, "50")
  CA.add_rule({"highway": "unclassified"}, "80")
  CA.add_rule({"highway": "residential"}, "40")
  CA.add_rule({"highway": "service"}, "40")
  CA.add_rule({"highway": "motorway_link"}, "90")
  CA.add_rule({"highway": "trunk_link"}, "80")
  CA.add_rule({"highway": "primary_link"}, "80")
  CA.add_rule({"highway": "secondary_link"}, "50")
  CA.add_rule({"highway": "tertiary_link"}, "50")
  CA.add_rule({"highway": "living_street"}, "20")


  """
  --------------------------------------------------
      DE - Germany
  --------------------------------------------------
  """
  DE = Country("DE")
  countries.append(DE)

  """ Default rules """
  DE.add_rule({"highway": "motorway"}, "none")
  DE.add_rule({"highway": "living_street"}, "10")
  DE.add_rule({"highway": "residential"}, "30")
  DE.add_rule({"zone:traffic": "DE:rural"}, "100")
  DE.add_rule({"zone:traffic": "DE:urban"}, "50")
  DE.add_rule({"zone:maxspeed": "DE:30"}, "30")
  DE.add_rule({"zone:maxspeed": "DE:urban"}, "50")
  DE.add_rule({"zone:maxspeed": "DE:rural"}, "100")
  DE.add_rule({"zone:maxspeed": "DE:motorway"}, "none")
  DE.add_rule({"bicycle_road": "yes"}, "30")
  

  """
  --------------------------------------------------
      EE - Estonia
  --------------------------------------------------
  """
  EE = Country("EE")
  countries.append(EE)

  """ Default rules """
  EE.add_rule({"highway": "motorway"}, "90")
  EE.add_rule({"highway": "trunk"}, "90")
  EE.add_rule({"highway": "primary"}, "90")
  EE.add_rule({"highway": "secondary"}, "50")
  EE.add_rule({"highway": "tertiary"}, "50")
  EE.add_rule({"highway": "unclassified"}, "90")
  EE.add_rule({"highway": "residential"}, "40")
  EE.add_rule({"highway": "service"}, "40")
  EE.add_rule({"highway": "motorway_link"}, "90")
  EE.add_rule({"highway": "trunk_link"}, "70")
  EE.add_rule({"highway": "primary_link"}, "70")
  EE.add_rule({"highway": "secondary_link"}, "50")
  EE.add_rule({"highway": "tertiary_link"}, "50")
  EE.add_rule({"highway": "living_street"}, "20")


  """ --- DO NOT MODIFY CODE BELOW THIS LINE --- """
  """ --- ADD YOUR COUNTRY OR STATE ABOVE --- """

  # Final step
  write_json(countries, filename)

def write_json(countries, filename = DEFAULT_OUTPUT_FILENAME):
  out_dict = {}
  for country in countries:
    out_dict.update(country.jsonify())
  json_string = json.dumps(out_dict, indent=2)
  with open(filename, "wb") as f:
    f.write(json_string)


class Region(object):
  ALLOWABLE_TAG_KEYS = ["highway", "zone:traffic", "bicycle_road", "zone:maxspeed"]
  ALLOWABLE_HIGHWAY_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "living_street"]
  def __init__(self, name):
    self.name = name
    self.rules = []

  def add_rule(self, tag_conditions, speed):
    new_rule = {}
    if not isinstance(tag_conditions, dict):
      raise TypeError("Rule tag conditions must be dictionary")
    if not all(tag_key in self.ALLOWABLE_TAG_KEYS for tag_key in tag_conditions):
      raise ValueError("Rule tag keys must be in allowable tag kesy") # If this is by mistake, please update ALLOWABLE_TAG_KEYS
    if 'highway' in tag_conditions:
      if not tag_conditions['highway'] in self.ALLOWABLE_HIGHWAY_TYPES:
        raise ValueError("Invalid Highway type {}".format(tag_conditions["highway"]))
    new_rule['tags'] = tag_conditions
    try:
      new_rule['speed'] = str(speed)
    except ValueError:
      raise ValueError("Rule speed must be string")
    self.rules.append(new_rule)

  def jsonify(self):
    ret_dict = {}
    ret_dict[self.name] = self.rules
    return ret_dict

class Country(Region):
  ALLOWABLE_COUNTRY_CODES = ["AF","AX","AL","DZ","AS","AD","AO","AI","AQ","AG","AR","AM","AW","AU","AT","AZ","BS","BH","BD","BB","BY","BE","BZ","BJ","BM","BT","BO","BQ","BA","BW","BV","BR","IO","BN","BG","BF","BI","KH","CM","CA","CV","KY","CF","TD","CL","CN","CX","CC","CO","KM","CG","CD","CK","CR","CI","HR","CU","CW","CY","CZ","DK","DJ","DM","DO","EC","EG","SV","GQ","ER","EE","ET","FK","FO","FJ","FI","FR","GF","PF","TF","GA","GM","GE","DE","GH","GI","GR","GL","GD","GP","GU","GT","GG","GN","GW","GY","HT","HM","VA","HN","HK","HU","IS","IN","ID","IR","IQ","IE","IM","IL","IT","JM","JP","JE","JO","KZ","KE","KI","KP","KR","KW","KG","LA","LV","LB","LS","LR","LY","LI","LT","LU","MO","MK","MG","MW","MY","MV","ML","MT","MH","MQ","MR","MU","YT","MX","FM","MD","MC","MN","ME","MS","MA","MZ","MM","NA","NR","NP","NL","NC","NZ","NI","NE","NG","NU","NF","MP","NO","OM","PK","PW","PS","PA","PG","PY","PE","PH","PN","PL","PT","PR","QA","RE","RO","RU","RW","BL","SH","KN","LC","MF","PM","VC","WS","SM","ST","SA","SN","RS","SC","SL","SG","SX","SK","SI","SB","SO","ZA","GS","SS","ES","LK","SD","SR","SJ","SZ","SE","CH","SY","TW","TJ","TZ","TH","TL","TG","TK","TO","TT","TN","TR","TM","TC","TV","UG","UA","AE","GB","US","UM","UY","UZ","VU","VE","VN","VG","VI","WF","EH","YE","ZM","ZW"]
  def __init__(self, ISO_3166_alpha_2):
    Region.__init__(self, ISO_3166_alpha_2)
    if ISO_3166_alpha_2 not in self.ALLOWABLE_COUNTRY_CODES:
      raise ValueError("Not valid IOS 3166 country code")
    self.regions = {}

  def add_region(self, name):
    self.regions[name] = Region(name)
    return self.regions[name]

  def jsonify(self):
    ret_dict = {}
    ret_dict[self.name] = {}
    for r_name, region in self.regions.items():
      ret_dict[self.name].update(region.jsonify())
    ret_dict[self.name]['Default'] = self.rules
    return ret_dict


if __name__ == '__main__':
  main()
