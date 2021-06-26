# Human Language Technologies Assignment 07
# Author: Mark Mondt mjm170130
import pickle
import nltk
import sys
import math
import random
import re
from stanza.server import CoreNLPClient

"""
params.txt format
0: Current User Type
1: Current User Name
2: Type of User's Response
3: User Response 
4: Is Bot Response a Prompt (T/F)
5: Bot Prompt (Coded)
    0 -> Confirmation
    1 -> Experience with the Franchise
    2 -> Favorite Game
    3 -> Favorite Character
    4 -> Curiosity Prompt
    5 -> Preference towards Entity
    6 -> Update
6: Prompt Arg
"""


def process_text(raw_text):
    mod_text = raw_text.lower()
    mod_text = mod_text.replace("--", " ")
    mod_text = "".join([char for char in mod_text if not re.match("[0-9]", char)])
    mod_text = "".join([char if not re.match("[().?!:;\\-,'\"]", char) else " " for char in mod_text])
    tokens = nltk.word_tokenize(mod_text)
    stop_words = nltk.corpus.stopwords.words("English")
    important_tokens = [t for t in tokens if t not in stop_words]

    return important_tokens


def update_knowledge(input_string, time, params):
    param_file = open(params, 'r', encoding='utf8')
    param_text = param_file.read()
    param_list = param_text.splitlines()
    param_file.close()

    corenlp_out = open("Junk.txt", 'w')

    # write name to a user file if it does not exist
    # if it does exist, remember that the user is returning
    if time == 0:
        user_file = open(input_string + "-facts.txt", 'a', encoding='utf8')
        user_file.close()
        user_file = open(input_string + "-facts.txt", 'r', encoding='utf8')
        user_text = user_file.read()
        user_list = user_text.splitlines()
        user_file.close()
        user_file = open(input_string + "-facts.txt", 'a', encoding='utf8')
        if len(user_list) < 1 or not user_list[0] == input_string:
            user_file.write(input_string + "\n")
            param_list[0] = "new"
        else:
            param_list[0] = "returning"
        param_list[1] = input_string
        user_file.close()
    else:
        user_file = open(param_list[1] + "-facts.txt", 'r', encoding='utf8')
        user_text = user_file.read()
        user_list = user_text.splitlines()
        user_file.close()
        user_file = open(param_list[1] + "-facts.txt", 'a', encoding='utf8')

        user_logs = open(param_list[1] + "-sents.txt", 'a', encoding='utf8')
        user_logs.close()
        user_logs = open(param_list[1] + "-sents.txt", 'r', encoding='utf8')
        logs_text = user_logs.read()
        logs_list = logs_text.splitlines()
        user_logs.close()
        user_logs = open(param_list[1] + "-sents.txt", 'a', encoding='utf8')

        # record new user input
        """
        processed_list = process_text(input_string)
        processed_string = ""
        for word in processed_list:
            processed_string += word + " "
        processed_string += "\n"
        """
        user_logs.write(input_string + "\n")

        # learn things about the user and decide question type
        if param_list[4] != "True":
            question_type = "None"

            tokens = nltk.word_tokenize(input_string.lower())
            pos_tags = nltk.pos_tag(tokens)

            if pos_tags[0][1][0] == "W" or (input_string.find('?') != -1 and
                                            (pos_tags[0][1][0] == "V" or pos_tags[0][1][0] == "M")):
                question_type = "Question"

                # perform dependency parse to find meaningful parts of the question
                subject = "-"
                relation = "-"
                subject_type = "subject"
                dependant = "-"
                with CoreNLPClient(annotators=["depparse"], be_quiet=True,
                                   stdout=corenlp_out, stderr=corenlp_out) as client:
                    annotation = client.annotate(input_string)
                    for edge in annotation.sentence[0].basicDependencies.edge:
                        if edge.dep == "nsubj":
                            subject = tokens[edge.target - 1]
                            relation = tokens[edge.source - 1]
                    for edge in annotation.sentence[0].basicDependencies.edge:
                        if edge.dep == "dep" and tokens[edge.source - 1] == relation:
                            dependant = tokens[edge.target - 1]
                        if edge.dep == "cop" and tokens[edge.source - 1] == relation:
                            relation = tokens[edge.target - 1]
                    if subject == "-":
                        for edge in annotation.sentence[0].basicDependencies.edge:
                            if edge.dep == "obl":
                                subject = tokens[edge.target - 1]
                                relation = tokens[edge.source - 1]
                        for edge in annotation.sentence[0].basicDependencies.edge:
                            if edge.dep == "dep" and tokens[edge.source - 1] == relation:
                                dependant = tokens[edge.target - 1]
                            if edge.dep == "cop" and tokens[edge.source - 1] == relation:
                                relation = tokens[edge.target - 1]
                    if subject == "-":
                        for edge in annotation.sentence[0].basicDependencies.edge:
                            if edge.dep == "obj":
                                subject_type = "object"
                                subject = tokens[edge.target - 1]
                                relation = tokens[edge.source - 1]
                        for edge in annotation.sentence[0].basicDependencies.edge:
                            if edge.dep == "dep" and tokens[edge.source - 1] == relation:
                                dependant = tokens[edge.target - 1]
                            if edge.dep == "cop" and tokens[edge.source - 1] == relation:
                                relation = tokens[edge.target - 1]

                param_list[3] = " ".join([subject, relation, subject_type, dependant])

            param_list[2] = question_type

        else:
            # perform sentiment analysis on prompt response, if necessary
            sentiment_codes = ["0", "1", "4", "5", "6"]
            if param_list[5] in sentiment_codes:
                param_list[3] = ""

                if input_string.lower().find("yes") != -1 \
                        or input_string.lower().find("sure") != -1 \
                        or input_string.lower().find("yep") != -1 \
                        or input_string.lower().find("affirm") != -1 \
                        or input_string.lower().find("roger") != -1 \
                        or input_string.lower().find("confirm") != -1:
                    param_list[3] = "Positive"

                if input_string.lower().find("nope") != -1:
                    param_list[3] = "Negative"

                if param_list[3] == "":
                    with CoreNLPClient(annotators=["sentiment"], be_quiet=True,
                                       stdout=corenlp_out, stderr=corenlp_out) as client:
                        annotation = client.annotate(input_string)
                    param_list[3] = annotation.sentence[0].sentiment
            else:
                param_list[3] = input_string

        user_logs.close()

    # write params back to the file
    new_text = ""
    for param in param_list:
        new_text += param + "\n"
    param_file = open(params, 'w', encoding='utf8')
    param_file.write(new_text)
    param_file.close()
    user_file.close()
    corenlp_out.close()


def generate_bot_string(time, params, knowledge):
    param_file = open(params, 'r', encoding='utf8')
    param_text = param_file.read()
    param_list = param_text.splitlines()
    param_file.close()

    user_file = open(param_list[1] + "-facts.txt", 'r', encoding='utf8')
    user_text = user_file.read()
    user_list = user_text.splitlines()
    user_file.close()
    user_file = open(param_list[1] + "-facts.txt", 'a', encoding='utf8')


    return_string = ""

    # Greet new or previous user
    if time == 0:
        if param_list[0] == "new":
            return_string = "Hello " + param_list[1] + ", ask me something you want to know about Castlevania"
        else:
            return_string = "Welcome back " + param_list[1] + \
                            ", ask me something you want to know about Castlevania"

        param_list[4] = "False"
    else:

        # Respond to a prompt
        if param_list[4] == "True":

            param_list[4] = "False"

            if param_list[5] == "6":
                if param_list[3] == "Positive" or param_list[3] == "Very Positive":
                    return_string += "Understood. "
                else:
                    return_string += "Interesting. "

                    if param_list[6] == "favoriteGame":
                        param_list[4] = "True"
                        param_list[5] = "2"
                        return_string += "What is your new favorite Castlevania game?"

                    elif param_list[6] == "favoriteCharacter":
                        param_list[4] = "True"
                        param_list[5] = "3"
                        return_string += "What is your new favorite Castlevania character?"

            # currently unused
            if param_list[5] == "0" and (param_list[3] == "Positive" or param_list[3] == "Very Positive"):
                return_string += param_list[6]

            if param_list[5] == "1":
                if param_list[3] == "Positive":
                    user_file.write("user experienced castlevania\n")
                    return_string += "Another cultured one! "
                else:
                    user_file.write("user experienced nothing\n")
                    return_string += "Then I shall teach you! "

            if param_list[5] == "2":
                line_no = line_in_list(user_list, "favoriteGame")

                # handle if fact already exists in file
                if line_no != -1:
                    user_list.pop(line_no)
                    user_file.close()
                    new_text = ""
                    for fact in user_list:
                        new_text += fact + "\n"
                    user_file = open(param_list[1] + "-facts.txt", 'w', encoding='utf8')
                    user_file.write(new_text)
                    user_file.close()
                    user_file = open(param_list[1] + "-facts.txt", 'a', encoding='utf8')
                else:
                    return_string += "Excellent choice! "

                user_file.write("user favoriteGame " + param_list[3] + "\n")

            if param_list[5] == "3":
                line_no = line_in_list(user_list, "favoriteCharacter")

                # handle if fact already exists in file
                if line_no != -1:
                    user_list.pop(line_no)
                    user_file.close()
                    new_text = ""
                    for fact in user_list:
                        new_text += fact + "\n"
                    user_file = open(param_list[1] + "-facts.txt", 'w', encoding='utf8')
                    user_file.write(new_text)
                    user_file.close()
                    user_file = open(param_list[1] + "-facts.txt", 'a', encoding='utf8')
                else:
                    return_string += "Excellent choice! "

                user_file.write("user favoriteCharacter " + param_list[3] + "\n")

            if param_list[5] == "4" and param_list[3] == "Positive":
                return_string += param_list[6]
            elif param_list[5] == "4":
                return_string += "Fair enough I suppose. "

            if param_list[5] == "5":
                if param_list[3] == "Positive":
                    user_file.write("user likes " + param_list[6] + "\n")
                    return_string += "I see! "
                elif param_list[3] == "Negative":
                    user_file.write("user dislikes " + param_list[6] + "\n")
                    return_string += "Well then. "
                else:
                    return_string += "I'm not sure how to take that. "

            if param_list[4] == "False":
                return_string += "I have much more to share. "

        # Answer user question when asked
        elif param_list[2] == "Question":
            question_code = param_list[3]
            question_list = question_code.split()
            return_string += generate_answer("", question_list[0], question_list[1], question_list[3], knowledge)

        # Prompt user if they haven't asked a question
        elif param_list[2] == "None":
            param_list[4] = "True"
            last_prompt = param_list[5]
            new_prompt = random.randint(1, 5)
            update_likelihood = random.randint(0, 2)

            if new_prompt == 1 and last_prompt != new_prompt and line_in_list(user_list, "experienced") != -1:
                param_list[5] = "1"
                return_string += "Tell me, have you ever experienced the Castlevania franchise? "

            elif new_prompt == 2 and last_prompt != new_prompt:
                # check for favoriteGame
                line_no = line_in_list(user_list, "favoriteGame")
                if line_no != -1 and update_likelihood == 2:
                    param_list[5] = "6"
                    param_list[6] = "favoriteGame"
                    return_string += "Is your favorite Castlevania game still"
                    return_string += user_list[line_no][
                                     user_list[line_no].find("favoriteGame") + len("favoriteGame"):]
                    return_string += "? "
                elif line_no == -1:
                    param_list[5] = "2"
                    return_string += "Tell me, what is your favorite Castlevania game? "
                else:
                    # Syncing files for curiosity prompt
                    new_text = ""
                    for param in param_list:
                        new_text += param + "\n"
                    param_file = open(params, 'w', encoding='utf8')
                    param_file.write(new_text)
                    param_file.close()

                    return_string += generate_curiosity_prompt(params, knowledge)

                    param_file = open(params, 'r', encoding='utf8')
                    param_text = param_file.read()
                    param_list = param_text.splitlines()
                    param_file.close()
                    param_list[5] = "4"

            elif new_prompt == 3 and last_prompt != new_prompt:
                # check for favoriteCharacter
                line_no = line_in_list(user_list, "favoriteCharacter")
                if line_no != -1 and update_likelihood == 2:
                    param_list[5] = "6"
                    param_list[6] = "favoriteCharacter"
                    return_string += "Is your favorite Castlevania character still"
                    return_string += user_list[line_no][
                                     user_list[line_no].find("favoriteCharacter") + len("favoriteCharacter"):]
                    return_string += "? "
                elif line_no == -1:
                    param_list[5] = "3"
                    return_string += "Tell me, what is your favorite Castlevania character? "
                else:
                    # Syncing files for curiosity prompt
                    new_text = ""
                    for param in param_list:
                        new_text += param + "\n"
                    param_file = open(params, 'w', encoding='utf8')
                    param_file.write(new_text)
                    param_file.close()

                    return_string += generate_curiosity_prompt(params, knowledge)

                    param_file = open(params, 'r', encoding='utf8')
                    param_text = param_file.read()
                    param_list = param_text.splitlines()
                    param_file.close()
                    param_list[5] = "4"

            elif new_prompt == 5 and last_prompt != new_prompt:
                # check level of knowledge on user
                user_logs = open(param_list[1] + "-facts.txt", 'r', encoding='utf8')
                logs_text = user_logs.read()
                logs_list = logs_text.splitlines()
                user_logs.close()

                valid = []
                if len(logs_list) > 20:
                    # prompt about frequently mentioned entity
                    top_list = tf_isf(param_list[1] + "-sents.txt")
                    likes_indices = lines_in_list(logs_list, "user likes")
                    dislikes_indices = lines_in_list(logs_list, "user dislikes")
                    likes_indices.extend(dislikes_indices)

                    valid = top_list.copy()
                    for entity in top_list:
                        for sent in likes_indices:
                            if logs_list[sent].lower().find(entity.lower()) != -1:
                                valid.remove(entity)

                if len(valid) == 0:
                    # prompt about unknown entity
                    topic_index = random.randint(0, len(knowledge.keys()) - 1)
                    topic = list(knowledge.keys())[topic_index]
                    subject_index = random.randint(0, len(knowledge[topic][0].keys()) - 1)
                    subject = list(knowledge[topic][0].keys())[subject_index]
                    return_string += "Tell me, how do you feel about " + subject + "? "
                    param_list[6] = subject
                    param_list[5] = "5"
                else:
                    return_string += "You seem interested in " + valid[0] + ", how do you feel about it? "
                    param_list[6] = valid[0]
                    param_list[5] = "5"

            else:
                # Syncing files for curiosity prompt
                new_text = ""
                for param in param_list:
                    new_text += param + "\n"
                param_file = open(params, 'w', encoding='utf8')
                param_file.write(new_text)
                param_file.close()

                return_string += generate_curiosity_prompt(params, knowledge)

                param_file = open(params, 'r', encoding='utf8')
                param_text = param_file.read()
                param_list = param_text.splitlines()
                param_file.close()
                param_list[5] = "4"

        # Default response
        else:
            return_string = "Sorry, I got lost in thought. Where were we?"
            param_list[4] = "False"

    # write params back to the file
    new_text = ""
    for param in param_list:
        new_text += param + "\n"
    param_file = open(params, 'w', encoding='utf8')
    param_file.write(new_text)
    param_file.close()
    return return_string + "\n"


def generate_curiosity_prompt(params, ie_info):
    param_file = open(params, 'r', encoding='utf8')
    param_text = param_file.read()
    param_list = param_text.splitlines()
    param_file.close()

    topic_index = random.randint(0, len(ie_info.keys()) - 1)
    topic = list(ie_info.keys())[topic_index]
    prompt_string = "Here's an interesting tome! Want to learn about "
    prompt_string += topic
    prompt_string += "?"

    param_list[6] = generate_answer(topic, topic, "", "", ie_info)

    # write params back to the file
    new_text = ""
    for param in param_list:
        new_text += param + "\n"
    param_file = open(params, 'w', encoding='utf8')
    param_file.write(new_text)
    param_file.close()
    return prompt_string


def generate_answer(key, entity, relation, relation_mod, ie_info):
    keys = ie_info.keys()
    if key.lower() in keys:
        keys = [key.lower()]
    role_index = 0

    return_strings = []
    pre_flavor = ""
    post_flavor = ""

    for key in keys:
        if entity.lower() in ie_info[key][role_index].keys():
            ie_objects = ie_info[key][role_index][entity.lower()]
            for ie_object in ie_objects:
                ie_parts = ie_object.split(';')
                if ie_parts[1].lower().find(relation.lower()) != -1:
                    return_strings.append(" ".join(ie_parts))
            pre_flavor = "Let's see, it says here \""
            post_flavor = "\". "

    # handle contingency that no entity matches were found
    if len(return_strings) == 0:
        best_similarity = sys.maxsize
        for key in keys:
            best_entries = []
            for entry in ie_info[key][role_index].keys():
                entry_similarity = calc_edit_dist(entry.lower(), entity.lower())
                if entry_similarity < best_similarity:
                    best_similarity = entry_similarity
                    best_entries = [entry]
                elif entry_similarity == best_similarity:
                    best_entries.append(entry)

            for entry in best_entries:
                ie_objects = ie_info[key][role_index][entry]
                for ie_object in ie_objects:
                    ie_parts = ie_object.split(';')
                    if ie_parts[1].lower().find(relation.lower()) != -1:
                        return_strings.append(" ".join(ie_parts))
        pre_flavor = "My Adamic is rusty, but this one looks like \""
        post_flavor = "\". "

    # find some fact for the closest subject
    if len(return_strings) == 0:
        best_similarity = sys.maxsize
        for key in keys:
            best_entries = []
            for entry in ie_info[key][role_index].keys():
                entry_similarity = calc_edit_dist(entry.lower(), entity.lower())
                if entry_similarity < best_similarity:
                    best_similarity = entry_similarity
                    best_entries = [entry]
                elif entry_similarity == best_similarity:
                    best_entries.append(entry)

            for entry in best_entries:
                ie_objects = ie_info[key][role_index][entry]
                for ie_object in ie_objects:
                    ie_parts = ie_object.split(';')
                    # maybe add way of finding more relevant relations
                    return_strings.append(" ".join(ie_parts))
        pre_flavor = "Well I'm not sure it's relevant, but \""
        post_flavor = "\". Hope that helped in some way. "

    chosen = -1
    max_length = 0
    for num, string in enumerate(return_strings):
        if string.find(relation_mod) != -1 and len(string) > max_length:
            chosen = num
            max_length = len(string)

    if chosen == -1:
        chosen = random.randint(0, len(return_strings) - 1)

    return pre_flavor + return_strings[chosen] + post_flavor


def line_in_list(input_list, phrase):
    line = -1
    for index, item in enumerate(input_list):
        if item.find(phrase) != -1:
            line = index
    return line


def lines_in_list(input_list, phrase):
    lines = []
    for index, item in enumerate(input_list):
        if item.find(phrase) != -1:
            lines.append(index)
    return lines


def calc_edit_dist(word1, word2):
    matrix = [[0 for x in range(len(word2))] for y in range(len(word1))]
    for k in range(len(word2)):
        matrix[0][k] = k
    for l in range(0, len(word1)):
        matrix[l][0] = l

    for i in range(1, len(word1)):
        for j in range(1, len(word2)):
            if word1[i] == word2[j]:
                sub_cost = 0
            else:
                sub_cost = 1
            delete_cost = matrix[i - 1][j] + 1
            insert_cost = matrix[i][j - 1] + 1
            sub_cost = matrix[i - 1][j - 1] + sub_cost
            matrix[i][j] = min(delete_cost, insert_cost, sub_cost)

    return matrix[len(word1) - 1][len(word2) - 1]


def tf_isf(filename):
    file = open(filename, 'r', encoding='utf8')
    file_text = file.read().lower()
    sent_list = file_text.splitlines()
    file.close()

    freq_dict = {}
    sents_per_term = {}
    total_tokens = 0
    for sent in sent_list:
        file = open(filename, 'r', encoding='utf8')
        mod_sent = "".join([char if not re.match("[().?!:;\\-,'\"]", char) else " " for char in sent])
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(["user", "favoritecharacter", "favoritegame", "likes", "dislikes", "experienced"])

        words = nltk.word_tokenize(mod_sent)
        words = [w for w in words if w.isalpha() and w not in stopwords]
        word_set = set(words)
        freq_dict = {w: words.count(w) for w in word_set}
        sents_per_term = {w: 1 if w not in sents_per_term else sents_per_term[w] + 1 for w in word_set}
        total_tokens += len(words)

    for t in freq_dict.keys():
        freq_dict[t] = freq_dict[t] / total_tokens
    for term in sents_per_term:
        sents_per_term[term] = math.log((len(sent_list) + 1) / (sents_per_term[term] + 1))

    isf_dict = {term: sents_per_term[term] * freq_dict[term] for term in freq_dict}
    return_set = []
    for num, k in enumerate(sorted(isf_dict, key=lambda k: isf_dict[k], reverse=True)):
        if num < 10:
            return_set.append(k)
        else:
            break

    return return_set


print("\n\nYou can end this chatbot by typing quit at any time\n")
bot_string = "Hi, this is SeekerBot, and I’m here to share my knowledge of Castlevania.\n"
bot_string += "To allow me to get to know you better, what's your name?\n"
iteration = 0
param_file_name = "params.txt"
temp = open(param_file_name, 'w', encoding='utf8')
for x in range(7):
    temp.write("NULL\n")
temp.close()

ie_dicts = pickle.load(open("ie_base.pickle", 'rb'))

user_string = input(bot_string)
while user_string != "quit" and user_string != "Quit":
    update_knowledge(user_string, iteration, param_file_name)
    bot_string = generate_bot_string(iteration, param_file_name, ie_dicts)
    user_string = input(bot_string)
    iteration += 1
