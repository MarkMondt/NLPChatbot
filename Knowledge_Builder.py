import pickle
from stanza.server import CoreNLPClient

proto_base = pickle.load(open("knowledge_base.pickle", 'rb'))
new_base = {}

for num, element in enumerate(proto_base):
    new_base[element] = [{}, {}, {}]
    element_string = " ".join(proto_base[element])
    with CoreNLPClient(annotators=["openie"], be_quiet=True) as client:
        annotation = client.annotate(element_string)
        for sent in annotation.sentence:
            for triple in sent.openieTriple:
                triple_string = triple.subject + ";"
                triple_string += triple.relation + ";"
                triple_string += triple.object
                if triple.subject in new_base[element][0].keys():
                    new_base[element][0][triple.subject].append(triple_string)
                else:
                    new_base[element][0][triple.subject] = [triple_string]
                if triple.relation in new_base[element][1].keys():
                    new_base[element][1][triple.relation].append(triple_string)
                else:
                    new_base[element][1][triple.relation] = [triple_string]
                if triple.object in new_base[element][2].keys():
                    new_base[element][2][triple.object].append(triple_string)
                else:
                    new_base[element][2][triple.object] = [triple_string]
                print("\t\tWorking on a triple")
            print("\tWorking on a sentence")
    print("Working on element " + str(num))

pickle.dump(new_base, open("ie_base.pickle", 'wb'))
