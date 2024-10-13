import instaloader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import joblib  # Pour sauvegarder et charger le modèle
import matplotlib.pyplot as plt
import os

# Chemin pour sauvegarder et charger le modèle
model_filename = "decision_tree_instagram_model.pkl"

# 1. Charger et comprendre les données
data = pd.read_csv('dataset.csv')  # Remplacez par le chemin vers votre fichier

# Afficher les premières lignes pour comprendre la structure
print("Aperçu des données :")
print(data.head())

# 2. Nettoyage et transformation des données
# Gérer les valeurs manquantes (s'il y en a)
data = data.dropna()

# Transformer les colonnes booléennes (0 ou 1) si nécessaire
bool_columns = ['is_private', 'is_business_account', 'has_external_url', 'is_fake']
for col in bool_columns:
    data[col] = data[col].astype(int)

# 5. Sélectionner et créer des caractéristiques (features)
# Ajout des variables manquantes : username_has_number, full_name_has_number
features = ['edge_followed_by', 'edge_follow', 'username_length', 'full_name_length',
            'is_private', 'is_business_account', 'has_external_url', 
            'username_has_number', 'full_name_has_number']

X = data[features]
y = data['is_fake']

# 6. Appliquer le Random Oversampling pour augmenter la classe minoritaire
real_accounts = data[data['is_fake'] == 0]
fake_accounts = data[data['is_fake'] == 1]

# Sur-échantillonner les comptes réels
real_accounts_oversampled = resample(real_accounts,
                                     replace=True,  # Autoriser les doublons
                                     n_samples=len(fake_accounts),  # Echantillonner jusqu'à égalité
                                     random_state=42)

# Combiner les comptes réels sur-échantillonnés et les faux comptes
balanced_data = pd.concat([real_accounts_oversampled, fake_accounts])

X_balanced = balanced_data[features]
y_balanced = balanced_data['is_fake']

# 7. Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# 8. Entraîner et sauvegarder le modèle seulement s'il n'est pas déjà sauvegardé
if not os.path.exists(model_filename):
    print("Entraînement du modèle...")
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    # Sauvegarder le modèle entraîné
    joblib.dump(tree_model, model_filename)
    print(f"Modèle sauvegardé sous {model_filename}.")
else:
    # Charger le modèle sauvegardé
    print(f"Chargement du modèle depuis {model_filename}...")
    tree_model = joblib.load(model_filename)

# Évaluer le modèle
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Précision (Accuracy) : {accuracy:.2f}")
print("Rapport de classification :")
print(classification_rep)


""" 10. Web scraping avec Instaloader pour tester un compte Instagram

def get_instagram_profile_data(username):
    L = instaloader.Instaloader()
    profile = instaloader.Profile.from_username(L.context, username)
    
    # Extraire les informations pertinentes (sans has_channel, has_guides, is_joined_recently)
    profile_data = {
        'edge_followed_by': profile.followers / 1000,  # Normaliser en milliers
        'edge_follow': profile.followees / 1000,  # Normaliser en milliers
        'username_length': len(profile.username),
        'full_name_length': len(profile.full_name),
        'is_private': int(profile.is_private),
        'is_business_account': int(profile.is_business_account),
        'has_external_url': int(profile.external_url is not None),
        'username_has_number': int(any(char.isdigit() for char in profile.username)),
        'full_name_has_number': int(any(char.isdigit() for char in profile.full_name))
    }
    
    return profile_data

# Fonction pour prédire si un compte est faux ou réel
def predict_fake_or_real(username):
    profile_data = get_instagram_profile_data(username)
    
    # Convertir les données dans un format compatible avec le modèle
    input_data = pd.DataFrame([profile_data])
    
    # Prédire avec le modèle chargé
    prediction = tree_model.predict(input_data)
    
    if prediction[0] == 1:
        return f"Le compte {username} est probablement un faux compte."
    else:
        return f"Le compte {username} est probablement un vrai compte."

# Demander à l'utilisateur de saisir un nom d'utilisateur Instagram
username_input = input("Veuillez entrer un nom d'utilisateur Instagram pour analyse : ")

# Appeler la fonction de prédiction
result = predict_fake_or_real(username_input)
print(result)"""