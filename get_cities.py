import geopy

from geopy.geocoders import Nominatim
geolocator = Nominatim()
# location = geolocator.reverse("52.509669, 13.376294")

def get_howmany_cities(in_name):
    
    for line in open(in_name):
        uid = line.strip().split(' ')[0]
        set_cities = set()
        coors = get_coordinates(line)
        for i, co in enumerate(coors):
            # print(i, co)
            location = geolocator.reverse(co)
            set_cities.add(location)
        print(uid, len(set_cities))


def get_coordinates(line):
    coors = line.strip().split("{'type': 'Point', 'coordinates': [")[1:]
    coors = [co[:-3] for co in coors]
    for i, co in enumerate(coors):
        print(i, co)
    # print(coors)
    return coors


get_howmany_cities('/Users/kay/EXP/come_on_user/personality/data/large_510_geo_-1.txt')