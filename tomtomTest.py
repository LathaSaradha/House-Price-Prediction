# import http.client
#
# conn = http.client.HTTPSConnection("rdrunnerxx-trackservice.p.rapidapi.com")
#
# payload = "{    \"StartLocation\":{       \"Address\":{          " \
#           "\"Street\":\"65-99-65-31 Booth St\",         \"City\":\"Rego Park\",       " \
#           "  \"State\":\"NY\",         \"PostalCode\":\"11374\",         \"Country\":\"US\"      }   }, " \
#           " \"FinishLocation\":{       \"Address\":{          \"Street\":\"Marine Park\",      " \
#           "   \"City\":\"Brooklyn\",         \"State\":\"NY\",         \"PostalCode\":\"\",    " \
#           "     \"Country\":\"US\"      }   },   \"DistanceUnit\":0}"
#
# headers = {
#     'x-rapidapi-host': "rdrunnerxx-trackservice.p.rapidapi.com",
#     'x-rapidapi-key': "",
#     'contenttype': "text/json; charset=utf-8",
#     'content-type': "application/json",
#     'accept': "application/json"
#     }
# conn.request("POST", "/distance", payload, headers)
#
# res = conn.getresponse()
# data = res.read()
#589d40092fmshf476844235b8cdbp197bc4jsnb0ff9569fca6
# print(data.decode("utf-8"))





import http.client

conn = http.client.HTTPSConnection("010pixel-distance-v1.p.rapidapi.com")

headers = {
    'x-rapidapi-host': "010pixel-distance-v1.p.rapidapi.com",
    'x-rapidapi-key': "589d40092fmshf476844235b8cdbp197bc4jsnb0ff9569fca6"
    }

conn.request("GET", "/?unit=K&lat1=10&long1=-25.3&lat2=34.5&long2=-403.4", headers=headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))



# def main():
#     print("Inside Main")
#
#
#
#
# if __name__ == '__main__':
#     main()