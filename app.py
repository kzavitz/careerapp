from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import nltk
from datetime import datetime

# For logging results into Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open("ChatLogs").sheet1

# NLTK setup
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english")) | {
    "hi", "interest", "pursue", "career", "passion", "hello", "there", "like", "mostly",
    "make", "good", "want", "play", "playing", "grade", "family", "toward", "enjoy",
    "going", "music", "would", "student", "work"
}

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/chat": {"origins": "*"}})

careers_df = pd.read_csv("SpreadsheetAUD.csv").fillna("")

# ---------------- Career avenues updated ----------------
CAREER_AVENUES = {
    "outdoors",
    "hands_on",
    "people",
    "culture",
    "data_analysis",
    "instruments_sound",
    "teaching",
    "technology"
}

FOLLOWUP_OPTION_TO_AVENUE = {
    "bringing people together": ["people"],
    "culture and tradition": ["culture"],
    "organizing or running events": ["people", "outdoors"],
    "working with your hands and fixing things": ["hands_on"],
    "technology and programming": ["technology"],
    "music and sound": ["instruments_sound"],
    "teaching or sharing historical knowledge": ["teaching"],
}

# ---- INTEREST TRANSLATION updated to split people_culture into people and culture ----
INTEREST_TRANSLATION = {
    "cars": {
        "pathways": ["hands_on", "technology", "instruments_sound"],
        "prompt": "When you say you like cars, what part do you enjoy most?",
        "options": [
            "Working with your hands and fixing things",
            "Design, aesthetics, or how things sound",
            "Technology and innovation",
            "Media, culture, or branding"
        ]
    },
    "food": {
        "pathways": ["people", "culture", "hands_on", "teaching"],
        "prompt": "What do you enjoy most about food?",
        "options": [
            "Cooking and creating",
            "Culture and tradition",
            "Bringing people together",
            "Organizing or running events"
        ]
    },
    "sports": {
        "pathways": ["hands_on", "people", "culture", "outdoors"],
        "prompt": "What do you enjoy most about sports?",
        "options": [
            "Physical performance",
            "Health and fitness",
            "Team environments",
            "Competition and strategy"
        ]
    },
    "games": {
        "pathways": ["technology", "instruments_sound", "data_analysis"],
        "prompt": "What do you like most about video games?",
        "options": [
            "Music and sound",
            "Technology and programming",
            "Storytelling and worlds",
            "Design and interaction"
        ]
    },
    "nature": {
        "pathways": ["outdoors", "hands_on", "people", "culture"],
        "prompt": "When you say you like nature, what part do you enjoy most?",
        "options": [
            "Exploring and being outdoors",
            "Protecting the environment",
            "Studying plants, animals, or ecosystems",
            "Photography or artistic representation"
        ]
    },
    "technology": {
        "pathways": ["technology", "data_analysis", "hands_on"],
        "prompt": "What do you enjoy most about technology?",
        "options": [
            "Programming and problem-solving",
            "Building or designing devices",
            "Exploring new innovations",
            "Understanding systems and data"
        ]
    },
    "history": {
        "pathways": ["teaching", "people", "culture", "data_analysis"],
        "prompt": "What part of history interests you most?",
        "options": [
            "Studying past events",
            "Understanding cultures",
            "Analyzing patterns over time",
            "Teaching or sharing historical knowledge"
        ]
    },
    "social_media": {
        "pathways": ["people", "culture", "technology", "data_analysis"],
        "prompt": "What do you enjoy most about social media?",
        "options": [
            "Creating content (videos, graphics, photos)",
            "Engaging with communities and followers",
            "Analyzing trends and insights",
            "Learning new tech tools and apps"
        ]
    },
    "youtube": {
        "pathways": ["technology", "people", "culture", "instruments_sound"],
        "prompt": "What do you enjoy about YouTube?",
        "options": [
            "Making videos and storytelling",
            "Editing and production",
            "Sharing music or sound content",
            "Building a personal brand or channel"
        ]
    },
    "tiktok": {
        "pathways": ["technology", "people", "culture", "instruments_sound"],
        "prompt": "What part of TikTok do you like most?",
        "options": [
            "Creating fun or viral content",
            "Music and sound editing",
            "Following trends and community engagement",
            "Using apps and tech creatively"
        ]
    },
    "content_creating": {
        "pathways": ["technology", "people", "culture", "instruments_sound"],
        "prompt": "What kind of content do you like to create?",
        "options": [
            "Videos or streams",
            "Music or audio projects",
            "Digital graphics and editing",
            "Teaching or sharing knowledge"
        ]
    },
    "kpop": {
        "pathways": ["people", "culture", "instruments_sound", "teaching"],
        "prompt": "What draws you to KPop?",
        "options": [
            "The music and sound production",
            "Performance and choreography",
            "Fan communities and culture",
            "Learning about music styles and language"
        ]
    },
    "rap": {
        "pathways": ["instruments_sound", "people", "culture", "teaching"],
        "prompt": "What do you enjoy most about rap?",
        "options": [
            "Writing lyrics and storytelling",
            "Rhythm and sound production",
            "Cultural expression and influence",
            "Sharing knowledge or teaching techniques"
        ]
    },
    "being_rich": {
        "pathways": ["people", "culture", "technology", "data_analysis"],
        "prompt": "When you say you want to be rich, what motivates you?",
        "options": [
            "Starting businesses or entrepreneurship",
            "Learning about finance and tech",
            "Influencing people and culture",
            "Creating products or music that earn money"
        ]
    },
    "eat": {
        "pathways": ["hands_on", "people", "culture", "teaching"],
        "prompt": "What part about eating do you enjoy most?",
        "options": [
            "Cooking and experimenting with flavors",
            "Sharing meals and social experiences",
            "Learning about different cultures through food",
            "Teaching others recipes or techniques"
        ]
    },
}

# ---------------- Helpers ----------------

def normalize_word(word):
    return stemmer.stem(lemmatizer.lemmatize(word))

def extract_keywords(user_input):
    words = re.findall(r'\b[a-zA-Z]+\b', user_input.lower())
    return [normalize_word(w) for w in words if w not in stop_words and len(w) > 3]

def detect_surface_interests(text):
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return [w for w in words if w in INTEREST_TRANSLATION]

# --- Text phrase → avenue mapping ---
TEXT_TO_AVENUE = {
    "hands": "hands_on",
    "hands-on": "hands_on",
    "working with hands": "hands_on",
    "working with my hands": "hands_on",
    "fixing things": "hands_on",
    "technology": "technology",
    "programming": "technology",
    "computers": "technology",
    "music": "instruments_sound",
    "sound": "instruments_sound",
    "playing instruments": "instruments_sound",
    "teaching": "teaching",
    "sharing knowledge": "teaching",
    "people": "people",
    "culture": "culture",
    "outdoors": "outdoors",
    "nature": "outdoors",
    "data": "data_analysis",
    "analysis": "data_analysis",
    "numbers": "data_analysis",
}

# --- Updated follow-up detection (fuzzy match) ---
def detect_followup_avenues(text):
    """
    Check if text contains any follow-up phrases and map to career avenues.
    This uses both exact phrase matches and keyword -> avenue mapping.
    """
    text_clean = text.lower().strip()
    avenues = set()

    # 1. Check original follow-up options (exact or substring match)
    for option, mapped in FOLLOWUP_OPTION_TO_AVENUE.items():
        if option in text_clean:
            avenues.update(mapped)

    # 2. Check keyword → avenue map
    for phrase, avenue in TEXT_TO_AVENUE.items():
        if phrase in text_clean:
            avenues.add(avenue)

    return list(avenues)

# --- Updated avenue detection ---
def detect_avenues(text, surface_interests):
    """
    Combines:
    - Direct avenues mentioned in the text (CAREER_AVENUES)
    - Surface interest pathways
    - Keyword/phrase → avenue mapping
    """
    avenues = set()
    words = set(re.findall(r"\b[a-zA-Z_]+\b", text.lower()))

    # 1. Direct CAREER_AVENUES matches
    for w in words:
        if w in CAREER_AVENUES:
            avenues.add(w)

    # 2. Surface interest → pathways
    for interest in surface_interests:
        for p in INTEREST_TRANSLATION[interest]["pathways"]:
            p_norm = p.replace("-", "_")
            if p_norm in CAREER_AVENUES:
                avenues.add(p_norm)

    # 3. Keyword/phrase → avenue map
    for phrase, avenue in TEXT_TO_AVENUE.items():
        if phrase in text.lower():
            avenues.add(avenue)

    return list(avenues)

def clean_description(description):
    sentences = re.split(r'(?<=[.!?])\s+', description.strip())
    return ' '.join(sentences[:3])

# ---------------- Routes ----------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    active_interest = request.json.get("active_interest")

    if not user_input:
        return jsonify({"message": "Please enter something about your interests!", "careers": [], "active_interest": None})

    if not active_interest:
        surface_interests = detect_surface_interests(user_input)
        if surface_interests:
            interest = surface_interests[0]
            meta = INTEREST_TRANSLATION[interest]
            message = (
                f"I see you mentioned <strong>{interest}</strong>.<br>"
                f"{meta['prompt']}<br>"
                + "<br>".join([f"• {opt}" for opt in meta["options"]])
            )
            return jsonify({"message": message, "careers": [], "active_interest": interest})

    keywords = extract_keywords(user_input)
    avenues = []

    if active_interest:
        avenues = detect_followup_avenues(user_input)
    else:
        surface_interests = detect_surface_interests(user_input)
        avenues = detect_avenues(user_input, surface_interests)

    matched_careers = []

    for _, row in careers_df.iterrows():
        csv_keywords = {normalize_word(k) for k in str(row["Keywords"]).lower().split(", ")}
        career_avenues = {a.strip() for a in str(row.get("Career Avenues", "")).split(",")}
        matched_keywords = list(set(keywords) & csv_keywords)
        matched_avenues = list(set(avenues) & career_avenues)
        if matched_keywords or matched_avenues:
            noc_code = str(row.get("NOC", "")).strip()

            noc_url = ""
            if noc_code and noc_code.lower() != "nan":
                noc_url = f"https://noc.esdc.gc.ca/Structure/NOCProfile?code={noc_code}&version=2021.0"


            matched_careers.append({
                "Job Title": row["Job Title"],
                "Description": clean_description(row["Description of the Job"]),
                "Link": row["Links"],
                "NOC": noc_code,
                "NOC_URL": noc_url,
                "Matched Keywords": matched_keywords,
                "Matched Avenues": matched_avenues,
                "score": len(matched_keywords) * 2 + len(matched_avenues)
            })

    matched_careers = sorted(matched_careers, key=lambda x: x["score"], reverse=True)[:3]

    timestamp = datetime.utcnow().isoformat()
    titles = ", ".join(c["Job Title"] for c in matched_careers) if matched_careers else "No matches"
    try:
        sheet.append_row([timestamp, user_input, titles])
    except Exception as e:
        print(e)

    if matched_careers:
        message = "Here are some career matches based on your response."
    else:
        message = (
            "That’s interesting! Can you tell me more about what you enjoy doing — "
            "for example, being outdoors, working with your hands, using technology, "
            "teaching others, working with sound and instruments, or exploring culture?"
        )

    return jsonify({"message": message, "careers": matched_careers, "active_interest": None})


if __name__ == "__main__":
    app.run(debug=True)
