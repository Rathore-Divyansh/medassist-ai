import json
import torch
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import random
from datetime import datetime
import numpy as np
import pandas as pd

from nnet import NeuralNet
from nltk_utils import bag_of_words
from flask import Flask, render_template, request, jsonify

# =========================================
# INITIAL SETUP
# =========================================

random.seed(datetime.now().timestamp())

device = torch.device('cpu')

# =========================================
# LOAD NLP MODEL
# =========================================

FILE = "models/data.pth"

model_data = torch.load(FILE)

input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']

nlp_model = NeuralNet(
    input_size,
    hidden_size,
    output_size
).to(device)

nlp_model.load_state_dict(model_state)

nlp_model.eval()

# =========================================
# LOAD DATASETS
# =========================================

diseases_description = pd.read_csv(
    "data/symptom_Description.csv"
)

diseases_description['Disease'] = (
    diseases_description['Disease']
    .str.lower()
    .str.strip()
)

disease_precaution = pd.read_csv(
    "data/symptom_precaution.csv"
)

disease_precaution['Disease'] = (
    disease_precaution['Disease']
    .str.lower()
    .str.strip()
)

symptom_severity = pd.read_csv(
    "data/Symptom-severity.csv"
)

symptom_severity = symptom_severity.map(
    lambda s:
    s.lower().strip().replace(" ", "_")
    if isinstance(s, str)
    else s
)

# Main disease dataset
dataset = pd.read_csv("data/dataset.csv")

# =========================================
# GLOBAL STORAGE
# =========================================

user_symptoms = set()

# =========================================
# FLASK APP
# =========================================

app = Flask(__name__)

# =========================================
# NLP SYMPTOM DETECTION
# =========================================

def get_symptom(sentence):

    sentence = nltk.word_tokenize(sentence)

    X = bag_of_words(sentence, all_words)

    X = X.reshape(1, X.shape[0])

    X = torch.from_numpy(X)

    output = nlp_model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)

    prob = probs[0][predicted.item()]

    prob = prob.item()

    return tag, prob

# =========================================
# HOME ROUTE
# =========================================

@app.route('/')
def index():

    data = []

    user_symptoms.clear()

    with open(
        "static/assets/files/ds_symptoms.txt",
        "r"
    ) as file:

        all_symptoms = file.readlines()

    for s in all_symptoms:

        cleaned = (
            s.replace("'", "")
            .replace("_", " ")
            .replace(",\n", "")
        )

        data.append(cleaned)

    data = json.dumps(data)

    return render_template(
        'index.html',
        data=data
    )

# =========================================
# CHAT ROUTE
# =========================================

@app.route('/symptom', methods=['GET', 'POST'])
print(request.json)
def predict_symptom():

    print("Request json:", request.json)

    sentence = request.json['sentence']

    cleaned_sentence = (
        sentence.replace(".", "")
        .replace("!", "")
        .lower()
        .strip()
    )

    # =====================================
    # FINAL DISEASE PREDICTION
    # =====================================

    if cleaned_sentence == "done":

        if not user_symptoms:

            response_sentence = random.choice([
                "I can't know what disease you may have if you don't enter any symptoms :)",
                "Meddy can't know the disease if there are no symptoms...",
                "You first have to enter some symptoms!"
            ])

        else:

            best_match_count = 0

            disease = "Unknown"

            normalized_user_symptoms = set()

            for symptom in user_symptoms:

                normalized_user_symptoms.add(
                    symptom.lower()
                    .strip()
                    .replace(" ", "_")
                )

            # =================================
            # FIND BEST MATCHING DISEASE
            # =================================

            for _, row in dataset.iterrows():

                symptoms = set()

                for col in dataset.columns[1:]:

                    value = row[col]

                    if pd.notna(value):

                        cleaned_symptom = (
                            str(value)
                            .lower()
                            .strip()
                            .replace(" ", "_")
                        )

                        symptoms.add(cleaned_symptom)

                matches = len(
                    normalized_user_symptoms.intersection(symptoms)
                )

                if matches > best_match_count:

                    best_match_count = matches

                    disease = row["Disease"]

            print("Predicted disease:", disease)

            # =================================
            # IF NO MATCH FOUND
            # =================================

            if disease == "Unknown":

                response_sentence = (
                    "Sorry, I could not determine the disease clearly. "
                    "Please enter more symptoms."
                )

                return jsonify(response_sentence)

            # =================================
            # DESCRIPTION
            # =================================

            description_row = diseases_description.loc[
                diseases_description['Disease']
                == disease.lower().strip()
            ]

            if not description_row.empty:

                description = (
                    description_row['Description']
                    .iloc[0]
                )

            else:

                description = "Description not available."

            # =================================
            # PRECAUTIONS
            # =================================

            precaution = disease_precaution[
                disease_precaution['Disease']
                == disease.lower().strip()
            ]

            if not precaution.empty:

                precautions = (
                    "Precautions: "
                    + str(precaution.Precaution_1.iloc[0]) + ", "
                    + str(precaution.Precaution_2.iloc[0]) + ", "
                    + str(precaution.Precaution_3.iloc[0]) + ", "
                    + str(precaution.Precaution_4.iloc[0])
                )

            else:

                precautions = "Precautions not available."

            # =================================
            # FINAL RESPONSE
            # =================================

            response_sentence = (
                "It looks to me like you have "
                + disease
                + ". <br><br>"
                + "<i>Description: "
                + description
                + "</i>"
                + "<br><br><b>"
                + precautions
                + "</b>"
            )

            # =================================
            # SEVERITY CHECK
            # =================================

            severity = []

            for each in normalized_user_symptoms:

                sev = symptom_severity.loc[
                    symptom_severity['Symptom']
                    == each,
                    'weight'
                ]

                if not sev.empty:

                    severity.append(sev.iloc[0])

            if severity:

                if (
                    np.mean(severity) > 4
                    or np.max(severity) > 5
                ):

                    response_sentence += (
                        "<br><br>"
                        "Considering your symptoms are severe, "
                        "you should consider talking to a doctor."
                    )

            user_symptoms.clear()

    # =====================================
    # SYMPTOM DETECTION
    # =====================================

    else:

        symptom, prob = get_symptom(sentence)

        print("Symptom:", symptom)
        print("Probability:", prob)

        if prob > 0.5:

            response_sentence = (
                f"Hmm, I'm {(prob * 100):.2f}% sure "
                f"this is {symptom.replace('_', ' ')}."
            )

            user_symptoms.add(symptom)

        else:

            response_sentence = (
                "I'm sorry, but I don't understand you."
            )

        print("Current user symptoms:", user_symptoms)

    return jsonify(response_sentence)

# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    app.run(debug=True)
