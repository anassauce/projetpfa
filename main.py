import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Charger votre dataset
df = pd.read_csv('dataset.csv')

# Afficher les premières lignes pour vérifier le chargement
print("Les premières lignes du dataset :")
print(df.head())

# Gérer les valeurs manquantes
df = df.fillna(df.mean())  # Remplacer les valeurs manquantes par la moyenne des colonnes

# Afficher les données après traitement des valeurs manquantes
print("Données après remplissage des valeurs manquantes :")
print(df.head())

# Normaliser les données si nécessaire
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Afficher les données normalisées
print("Données normalisées :")
print(df_scaled[:5])  # Afficher les 5 premières lignes normalisées

# Séparer les données en comptes réels et faux
real_accounts = df[df['is_fake'] == 0]
fake_accounts = df[df['is_fake'] == 1]

# Liste des variables à analyser
variables = ['edge_followed_by', 'edge_follow', 'username_length', 'full_name_length', 'is_private', 
             'is_joined_recently', 'has_channel', 'is_business_account', 'has_external_url']

# Créer une figure pour les distributions de chaque variable
plt.figure(figsize=(15, 10))

# Parcourir les variables et créer des sous-graphes
for i, var in enumerate(variables):
    plt.subplot(3, 3, i+1)
    sns.histplot(real_accounts[var], label='Real', color='blue', kde=True, stat="density")
    sns.histplot(fake_accounts[var], label='Fake', color='red', kde=True, stat="density")
    plt.legend()
    plt.title(f'Distribution de {var}')

# Ajuster les espacements entre les sous-graphes
plt.tight_layout()
plt.show()

# Calcul des corrélations entre les variables et l'indicateur de faux comptes
correlation_matrix = df.corr()

# Visualiser la matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matrice de corrélation des variables")
plt.show()
