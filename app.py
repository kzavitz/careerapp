from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from datetime import datetime


# Google Sheets Logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set up Sheets auth
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open("ChatLogs").sheet1  # Ensure your sheet exists and is shared with service account

# NLTK setup
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english")) | {
    "hi", "interest", "pursue", "career", "passion", "hello", "there", "like", "mostly", "make", "good", "want",
    "play", "playing", "grade", "family", "toward", "enjoy", "going", "music", "would", "student", "work"
}

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/chat": {"origins": "*"}})


careers_df = pd.read_csv("SpreadsheetAUD.csv").fillna("")

def normalize_word(word):
    return stemmer.stem(lemmatizer.lemmatize(word))

def extract_keywords(user_input):
    words = re.findall(r'\b[a-zA-Z]+\b', user_input.lower())
    keywords = set()
    for word in words:
        if word not in stop_words and len(word) > 3:
            keywords.add(normalize_word(word))
    return list(keywords)

def extract_important_keywords():
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(careers_df["Description of the Job"].astype(str))
    return vectorizer, tfidf_matrix, vectorizer.get_feature_names_out()

vectorizer, tfidf_matrix, tfidf_features = extract_important_keywords()

def get_top_keywords(index, num_keywords=40):
    row = tfidf_matrix[index].toarray()[0]
    top_indices = row.argsort()[-num_keywords:][::-1]
    return [tfidf_features[i] for i in top_indices]

def clean_description(description):
    sentences = re.split(r'(?<=[.!?])\s+', description.strip())
    if sentences and "?" in sentences[0]:
        sentences.pop(0)
    return ' '.join(sentences[:3])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"message": "Please enter something about your interests!", "careers": []})

    keywords = extract_keywords(user_input)
    matched_careers = []

    for idx, row in careers_df.iterrows():
        csv_keywords = {normalize_word(k) for k in str(row["Keywords"]).lower().split(", ")}
        matched_keywords = list(set(keywords) & csv_keywords)

        if matched_keywords:
            noc_code = str(row.get("NOC", "11111")).strip()
            noc_link = f"https://noc.esdc.gc.ca/Structure/NOCProfile?code={noc_code}&version=2021.0" if noc_code else "nocod"

            same_category_jobs = careers_df[careers_df["Category"] == row["Category"]][["Job Title", "Links"]].to_dict(orient="records")

            matched_careers.append({
                "Job Title": row["Job Title"],
                "Description": clean_description(row["Description of the Job"]),
                "Link": row["Links"],
                "Category": row["Category"],
                "NOC": noc_code,
                "NOC Link": noc_link,
                "Matched Keywords": matched_keywords,
                "Related Careers": same_category_jobs
            })

    matched_careers = sorted(matched_careers, key=lambda x: len(x["Matched Keywords"]), reverse=True)[:3]

    # Log to Google Sheet
    timestamp = datetime.utcnow().isoformat()
    matched_titles = ", ".join([job["Job Title"] for job in matched_careers]) if matched_careers else "No matches"
    try:
        sheet.append_row([timestamp, user_input, matched_titles])
    except Exception as e:
        print(f"Error logging to Google Sheets: {e}")

    if not matched_careers:
        return jsonify({
            "message": "I'm not seeing any matches yet. Tell me more about you and the things you like to do!",
            "careers": []
        })

    return jsonify({
        "keywords": keywords,
        "message": "Here are up to three career options you might be interested in!",
        "careers": matched_careers
    })

if __name__ == "__main__":
    app.run(debug=True)
