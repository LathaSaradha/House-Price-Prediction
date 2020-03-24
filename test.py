

from geopy import Nominatim
from geopy.extra.rate_limiter import RateLimiter
class test:

    def find_latitude_long(self):
        locator = Nominatim(user_agent="myGeocoder")
        geocode = RateLimiter(locator.geocode, min_delay_seconds=5)
        location = locator.geocode("94-41 43rd Avenue New York, NY")
        print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))


def main():
    print("inside Main")
    obj = test()

    obj.find_latitude_long()



if __name__ == '__main__':
    main()


