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
    "hi", "interest", "pursue", "career", "passion", "hello", "there",
    "like", "mostly", "make", "good", "want", "play", "playing",
    "grade", "family", "toward", "enjoy", "going", "music",
    "would", "student", "work"
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
    "technology",
    "sports",
    "health"
}


FOLLOWUP_OPTION_TO_AVENUE = {

    # SPORTS

    "physical performance": [
        "sports",
        "outdoors"
    ],

    "health and fitness": [
        "health",

    ],

    "team environments": [
        "sports",
        "people"
    ],

    "competition and strategy": [
        "sports",
        "data_analysis"
    ],

    "working with a team": [
        "sports",
        "people"
    ],

    # TECHNOLOGY

    "programming and problem-solving": [
        "technology",
        "data_analysis"
    ],

    "building or designing devices": [
        "technology",
        "hands_on"
    ],

    "exploring new innovations": [
        "technology"
    ],

    "understanding systems and data": [
        "technology",
        "data_analysis"
    ],

    # HISTORY

    "studying past events": [
        "culture",
        "teaching"
    ],

    "understanding cultures": [
        "culture",
        "people"
    ],

    "analyzing patterns": [
        "data_analysis"
    ],

    "teaching others": [
        "teaching",
        "people"
    ],

    # FOOD

    "cooking and creating": [
        "hands_on"
    ],

    "culture and tradition": [
        "culture"
    ],

    "bringing people together": [
        "people"
    ],

    "organizing or running events": [
        "people",
        "outdoors"
    ],

    # GAMES

    "music and sound": [
        "instruments_sound"
    ],

    "competition": [
        "sports"
    ],

    "design and worlds": [
        "technology",
        "culture"
    ]
}

INTEREST_TRANSLATION = {

    "sports": {
        "pathways": [
            "sports",
            "hands_on",
            "people",
            "culture",
            "outdoors"
        ],
        "prompt": "What do you enjoy most about sports?",
        "options": [
            "Physical performance",
            "Health and fitness",
            "Team environments",
            "Competition and strategy"
        ]
    },

    "technology": {
        "pathways": [
            "technology",
            "data_analysis",
            "hands_on"
        ],
        "prompt": "What do you enjoy most about technology?",
        "options": [
            "Programming and problem-solving",
            "Building or designing devices",
            "Exploring new innovations",
            "Understanding systems and data"
        ]
    },

    "food": {
        "pathways": [
            "people",
            "culture",
            "hands_on",
            "teaching"
        ],
        "prompt": "What do you enjoy most about food?",
        "options": [
            "Cooking and creating",
            "Culture and tradition",
            "Bringing people together",
            "Organizing or running events"
        ]
    },

    "games": {
        "pathways": [
            "technology",
            "instruments_sound",
            "data_analysis"
        ],
        "prompt": "What part of games interests you most?",
        "options": [
            "Programming and problem-solving",
            "Music and sound",
            "Competition",
            "Design and worlds"
        ]
    },

    "history": {
        "pathways": [
            "teaching",
            "people",
            "culture",
            "data_analysis"
        ],
        "prompt": "What part of history interests you most?",
        "options": [
            "Studying past events",
            "Understanding cultures",
            "Analyzing patterns",
            "Teaching others"
        ]
    }
}

# ---------------- Helpers ----------------

def normalize_word(word):
    return stemmer.stem(lemmatizer.lemmatize(word))

def extract_keywords(user_input):

    words = re.findall(
        r'\b[a-zA-Z]+\b',
        user_input.lower()
    )

    return [

        normalize_word(w)

        for w in words

        if w not in stop_words

        and len(w) > 3
    ]

def detect_surface_interests(text):

    words = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            text.lower()
        )
    )

    return [

        w

        for w in words

        if w in INTEREST_TRANSLATION
    ]


TEXT_TO_AVENUE = {

    "hands": "hands_on",
    "hands-on": "hands_on",
    "working with hands": "hands_on",
    "working with my hands": "hands_on",
    "fixing": "hands_on",

    "technology": "technology",
    "programming": "technology",
    "computers": "technology",

    "music": "instruments_sound",
    "sound": "instruments_sound",

    "teaching": "teaching",

    "sharing": "teaching",

    "people": "people",

    "culture": "culture",

    "outdoors": "outdoors",

    "outside": "outdoors",

    "nature": "outdoors",

    "data": "data_analysis",

    "analysis": "data_analysis",

    "numbers": "data_analysis",

    "hockey": "sports",
    "soccer": "sports",
    "basketball": "sports",
    "baseball": "sports",
    "football": "sports",

    "patterns": "data_analysis",
    "strategy": "data_analysis",
    "fitness": "sports",
    "innovation": "technology",
    "devices": "hands_on",
    "cultures": "culture",
    "events": "people",
    "systems": "technology",

    "acoustics": "instruments_sound",
    "acoustic": "instruments_sound",

    "cardiology": "health",
    "neurology": "health",
    "medicine": "health",
    "doctor": "health",
    "health": "health",
    "therapy": "health",
    "hospital": "health",

    # add weak sports recognition

    "blue jays": "sports",
    "leafs": "sports",
    "raptors": "sports",
    "nhl": "sports",
    "nba": "sports",
    "mlb": "sports",
    "nfl": "sports"
}

def detect_followup_avenues(text):

    text_clean = text.lower().strip()

    avenues = set()

    for option, mapped in FOLLOWUP_OPTION_TO_AVENUE.items():

        if option in text_clean:

            avenues.update(mapped)

    for phrase, avenue in TEXT_TO_AVENUE.items():

        if phrase in text_clean:

            avenues.add(avenue)

    return list(avenues)

def detect_avenues(text, surface_interests):

    avenues = set()

    words = set(
        re.findall(
            r"\b[a-zA-Z_]+\b",
            text.lower()
        )
    )

    for w in words:

        if w in CAREER_AVENUES:

            avenues.add(w)

    for interest in surface_interests:

        for p in INTEREST_TRANSLATION[interest]["pathways"]:

            p_norm = p.replace("-", "_")

            if p_norm in CAREER_AVENUES:

                avenues.add(p_norm)

    for phrase, avenue in TEXT_TO_AVENUE.items():

        if phrase in text.lower():

            avenues.add(avenue)

    return list(avenues)

def clean_description(description):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        description.strip()
    )

    return " ".join(sentences[:3])

# -------------------------------------------------
# WEAK MATCHING FUNCTION
# -------------------------------------------------

def weak_match_careers(user_input):

    text = user_input.lower()

    weak_matches = []

    words = re.findall(

        r"\b[a-zA-Z]+\b",

        text
    )

    for _, row in careers_df.iterrows():

        score = 0

        title = str(

            row["Job Title"]

        ).lower()

        description = str(

            row["Description of the Job"]

        ).lower()

        keywords = str(

            row["Keywords"]

        ).lower()

        for word in words:

            if len(word) < 4:

                continue

            if word in title:

                score += 3

            if word in keywords:

                score += 2

            if word in description:

                score += 1

        if score >= 2:

            noc_code = str(

                row.get(

                    "NOC",

                    ""

                )

            ).strip()

            noc_url = ""

            if noc_code and noc_code.lower() != "nan":

                noc_url = (

                    f"https://noc.esdc.gc.ca/"

                    f"Structure/NOCProfile?"

                    f"code={noc_code}"

                    f"&version=2021.0"
                )

            weak_matches.append({

                "Job Title":
                    row["Job Title"],

                "Description":
                    clean_description(
                        row[
                            "Description of the Job"
                        ]
                    ),

                "Link":
                    row["Links"],

                "NOC":
                    noc_code,

                "NOC_URL":
                    noc_url,

                "Matched Keywords":
                    [],

                "Matched Avenues":
                    ["weak_match"],

                "score":
                    score
            })

    weak_matches = sorted(

        weak_matches,

        key=lambda x: x["score"],

        reverse=True

    )

    return weak_matches

# -------------------------------------------------
# NEW WEAK MATCH FUNCTION
# -------------------------------------------------

def diversify_matches(matches, limit=3):

    if len(matches) <= limit:
        return matches

    selected = []

    used_reasons = set()

    # First pass:
    # prioritize unique match reasons

    for career in matches:

        reasons = set(
            career["Matched Keywords"]
            +
            career["Matched Avenues"]
        )

        unseen = reasons - used_reasons

        if unseen:

            selected.append(career)

            used_reasons.update(unseen)

        if len(selected) >= limit:

            return selected

    # Second pass:
    # fill remaining slots

    for career in matches:

        if career not in selected:

            selected.append(career)

        if len(selected) >= limit:

            break

    return selected

# ---------------- Routes ----------------

@app.route("/")
def index():

    return render_template("index.html")

@app.route("/chat", methods=["POST"])

def chat():

    user_input = request.json.get(
        "message",
        ""
    ).strip()

    active_interest = request.json.get(
        "active_interest"
    )

    if not user_input:

        return jsonify({

            "message":
            "Please enter something about your interests!",

            "careers": [],

            "active_interest": None
        })

    if not active_interest:

        surface_interests = detect_surface_interests(
            user_input
        )

        if surface_interests:

            interest = surface_interests[0]

            meta = INTEREST_TRANSLATION[
                interest
            ]

            message = (

                f"I see you mentioned "

                f"<strong>{interest}</strong>.<br>"

                f"{meta['prompt']}<br>"

                +

                "<br>".join(

                    [f"• {opt}"

                     for opt

                     in meta["options"]]
                )
            )

            return jsonify({

                "message": message,

                "careers": [],

                "active_interest": interest
            })

    keywords = extract_keywords(
        user_input
    )

    avenues = []

    if active_interest:

        avenues = detect_followup_avenues(
            user_input
        )

    else:

        surface_interests = detect_surface_interests(
            user_input
        )

        avenues = detect_avenues(

            user_input,

            surface_interests
        )

    matched_careers = []

    match_lines = []

    all_matched_keywords = set()

    all_matched_avenues = set()

    for _, row in careers_df.iterrows():

        csv_keywords = {

            normalize_word(k)

            for k in str(
                row["Keywords"]
            ).lower().split(", ")
        }

        career_avenues = {

            a.strip()

            for a in str(
                row.get(
                    "Career Avenues",
                    ""
                )
            ).split(",")
        }

        # NEW: expand combined avenue labels

        if "people_culture" in career_avenues:
            career_avenues.add(
                "people"
            )

            career_avenues.add(
                "culture"
            )

        matched_keywords = list(

            set(keywords)

            & csv_keywords
        )

        matched_keywords = list(

            set(keywords)

            & csv_keywords
        )

        matched_avenues = list(

            set(avenues)

            & career_avenues
        )

        if matched_keywords or matched_avenues:
            all_matched_keywords.update(
                matched_keywords
            )

            all_matched_avenues.update(
                matched_avenues
            )
            noc_code = str(
                row.get(
                    "NOC",
                    ""
                )
            ).strip()

            noc_url = ""

            if noc_code and noc_code.lower() != "nan":

                noc_url = (

                    f"https://noc.esdc.gc.ca/"

                    f"Structure/NOCProfile?"

                    f"code={noc_code}"

                    f"&version=2021.0"
                )

            matched_careers.append({

                "Job Title":
                row["Job Title"],

                "Description":
                clean_description(
                    row[
                        "Description of the Job"
                    ]
                ),

                "Link":
                row["Links"],

                "NOC":
                noc_code,

                "NOC_URL":
                noc_url,

                "Matched Keywords":
                matched_keywords,

                "Matched Avenues":
                matched_avenues,

                "score":

                len(matched_keywords) * 2

                +

                len(matched_avenues)
            })

    matched_careers = sorted(

        matched_careers,

        key=lambda x: x["score"],

        reverse=True

    )

    all_matches = matched_careers.copy()

    matched_careers = diversify_matches(

        matched_careers,

        limit=3
    )

    # ---------------- NEW FALLBACK ----------------

    if not matched_careers:

        matched_careers = weak_match_careers(
            user_input
        )

        if matched_careers:

            message = (

                "I couldn't find strong matches, "

                "but these careers share some "

                "overlapping ideas with what "

                "you mentioned. These are "

                "weaker suggestions."
            )

        else:

            message = (

                "Tell me a little more about "

                "what you enjoy doing so I can "

                "find better matches."
            )

    else:

        match_lines = []

        if all_matches:

            for i, career in enumerate(

                    all_matches,

                    start=1
            ):
                match_lines.append(

                    f"{i}) "

                    f"<a href='{career['Link']}' "

                    f"target='_blank'>"

                    f"{career['Job Title']}"

                    f"</a>"
                )

        input_terms = []

        input_terms.extend(
            keywords
        )

        input_terms.extend(
            avenues
        )

        input_terms = list(
            set(input_terms)
        )

        input_text = ", ".join(

            [f'"{x}"' for x in input_terms]

        ) if input_terms else "your interests"

        matched_terms = list(

            all_matched_keywords

        ) + list(

            all_matched_avenues
        )

        matched_text = ", ".join(

            [f'{x}' for x in matched_terms]

        ) if matched_terms else "general overlaps"

        message = (

            "Thanks for your response! "


            f"I found matches connected to: "

            f"{matched_text}.<br><br>"

            f"Here are some careers that may fit:"
        )

    timestamp = datetime.utcnow().isoformat()

    titles = ", ".join(

        c["Job Title"]

        for c in matched_careers

    ) if matched_careers else "No matches"

    try:

        sheet.append_row([

            timestamp,

            user_input,

            titles
        ])

    except Exception as e:

        print(e)

    return jsonify({

        "message":
            message,

        "careers":
            matched_careers,

        "all_matches":
            match_lines,

        "active_interest":
            None
    })

if __name__ == "__main__":

    app.run(debug=True)