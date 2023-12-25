
#Importation des librairies
import streamlit as st
from CBAO_Dash import dashboard_users
st.set_page_config(layout="wide")


#Chargement de l'image en arrière-plan
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url(https://i.ibb.co/Dkf4pYz/c4c29214-c614-4fab-af24-535c0f914889.gif);
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: no-fixed;
height: 100vh;
margin: 0;
display: flex;


}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown('<div style="text-align:center;width:100%;"><h1 style="color:black;background-color:#f7a900;border:#fc1c24;border-style:solid;border-radius:5px;">TABLEAU DE BORD INTERACTIF DE LA BOITE A IDEES DIGITALE</h1></div>', unsafe_allow_html=True)

#Espacement
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")

# Dictionnaire contenant les informations d'identification des utilisateurs autorisés
users_credentials = {
    'cbao-qualité': {'password': '2023', 'Zone': 'cbao-qualité'},
    'agence-pompidou': {'password': 'pompidou', 'Zone': 'Dakar centre'},
}

# Fonction pour vérifier les informations d'identification
def authenticate(username, password):
    if username in users_credentials and password == users_credentials[username]['password']:
        return True, users_credentials[username]['Zone']
    return False, None

# Page de login
with st.expander('LOGIN'):
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:

        # Formulaire de connexion
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type='password')
        login_button = st.button("Se connecter")

        if login_button:
            # Vérifier les informations d'identification
            authenticated, Zone = authenticate(username, password)

            if authenticated:
                st.session_state.authenticated = True
                st.session_state.Zone = Zone
                st.success(f"Connecté en tant que {username} (Zone: {Zone})")
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")

# ...

if st.session_state.authenticated:
    dashboard_users(st.session_state.Zone)

