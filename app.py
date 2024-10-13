from flask import Flask, render_template, request
import instaloader
import pandas as pd
import joblib

# Charger le modèle préentraîné depuis main.py
model_filename = "decision_tree_instagram_model.pkl"
tree_model = joblib.load(model_filename)

# Initialiser Flask
app = Flask(__name__)

# Fonction pour extraire les données du profil Instagram
def get_instagram_profile_data(username):
    L = instaloader.Instaloader()
    
    profile = instaloader.Profile.from_username(L.context, username)
    
    # Extraire les informations pertinentes
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

# Route pour la page d'accueil
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Récupérer le nom d'utilisateur entré par l'utilisateur
        username = request.form['username']
        
        # Extraire les données du profil Instagram
        try:
            profile_data = get_instagram_profile_data(username)
            
            # Convertir les données dans un format compatible avec le modèle
            input_data = pd.DataFrame([profile_data])
            
            # Faire la prédiction
            prediction = tree_model.predict(input_data)
            
            # Résultat de la prédiction
            if prediction[0] == 1:
                result = f"Le compte {username} est probablement un faux compte."
            else:
                result = f"Le compte {username} est probablement un vrai compte."
        
        except Exception as e:
            # En cas d'erreur (par exemple si le compte n'existe pas)
            result = f"Erreur : {str(e)}"
    
    return render_template('index.html', result=result)

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
