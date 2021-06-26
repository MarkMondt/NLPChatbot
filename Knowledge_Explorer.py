
import pickle
from stanza.server import CoreNLPClient

ie_info = pickle.load(open("ie_base.pickle", 'rb'))
new_base = {}

for key in ie_info.keys():
    for dict in ie_info[key]:
        for list in dict[list]:
            for element in list:
                print(element)
