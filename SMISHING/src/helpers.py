from nltk.tokenize import word_tokenize
import string, joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

    
def cleaner(text):
    
    tokens = word_tokenize(text)
    
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    # stemming of words
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
#     print(stemmed[:100])

    
    
    return stemmed#' '.join(stemmed)


 # load model
model = joblib.load("../save_models/svm.pkl")
model_vec = joblib.load("../save_models/count_vectorizer.pkl")
# model_scaler = joblib.load("../save_models/robustscaler.pkl")
model_encoder = joblib.load("../save_models/labelencoder.pkl")

def prep_4_model(t):
    t = ' '.join(cleaner(t))
    
    return model_vec.transform([t]).toarray()

def to_label(t):
    # print(t)
    return list(model_encoder.inverse_transform([t]))[0]


def make_predictions(text):

    search = prep_4_model(text)
    pred = model.predict(search)
    pred_proba = round(model.predict_proba(search)[0][0], 4)
    print(pred_proba)

    return to_label(pred.item()),pred_proba