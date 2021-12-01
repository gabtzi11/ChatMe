import nltk
from nltk.stem import WordNetLemmatizer

# Uncomment these downloaders only for the 1st run.
# nltk.download('popular', quiet=True)
# nltk.download('punkt')
# nltk.download('wordnet')

import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Read corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenisation
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Processing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keywords
GREETING_INPUTS = ("hello", "hi", "greetings", "hey", "yo")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello"]
BOOKING_ALERT = ("booking", "reservation", "flight", "travel", "ticket", "trip", "fly", "airplane", "plane", "book")
YEARS = ("2022", "22", "23", "2023")
MONTHS = (
"Jan", "January", "Feb", "February", "Mar", "March", "Apr", "April", "May", "May", "Jun", "June", "Jul", "July", "Aug",
"August", "Sep", "September", "Oct", "October", "Nov", "November", "Dec", "December")


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def booking(sentence):
    for word in sentence.split():
        if word.lower() in BOOKING_ALERT:
            return "Welcome to my flight booking system. "


def year(sentence):
    for word in sentence.split():
        if word in YEARS:
            return "Please enter the year you want to fly! I am currently taking booking from January 2022 to December 2023"


def month(sentence):
    for word in sentence.split():
        if word in MONTHS:
            return "Please enter the year you want to fly! I am currently taking booking from January 2022 to December 2023"


# Generating response
def response(user_response):
    gab_bot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        gab_bot_response = gab_bot_response + "I am sorry! I don't understand you"
        return gab_bot_response
    else:
        gab_bot_response = gab_bot_response + sent_tokens[idx]
        return gab_bot_response


flag = True
bookBool = False
yearBool = False
print("Gab-bot: My name is Gab-bot. You can call me Gab or just bot.")
while flag == True:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("Gab-bot: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("Gab-bot: " + greeting(user_response))

            elif booking(user_response) is not None:
                print("Gab-bot: " + booking(user_response))
                print("Gab-bot: Please enter the year you want to fly")
                bookBool = True

            elif bookBool and year(user_response) is not None:
                if user_response == "22":
                    yearS = "2022"
                elif user_response == "23":
                    yearS = "2023"
                else:
                    yearS = user_response
                print("Gab-bot: Which month of " + yearS + " do you want to fly?")
                yearBool = True
                bookBool = False

            elif yearBool and month(user_response) is not None:
                if user_response == "Jan" or "January":
                    monthS = "January"
                print("Gab-bot: " + monthS)

            else:
                print("Gab-bot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Gab-bot: Bye! See you in a bit!")
