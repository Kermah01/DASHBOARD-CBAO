
#Importation des librairies
import streamlit as st
from dashboard import dashboard_users, TOUTES_LES_ZONES
st.set_page_config(page_title="Dashboard Qualité de Service", layout="wide")


#Chargement de l'image en arrière-plan
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] [data-testid="stMain"] {{
background-image: url(https://i.ibb.co/Dkf4pYz/c4c29214-c614-4fab-af24-535c0f914889.gif);
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: scroll;
height: 100vh;
height: 100dvh;
margin: 0;
display: flex;


}}

/* Lien d'ancre des titres : hors du flux comme dans la version d'origine (évite un retour à la ligne) */
.titre-accueil [data-testid="stHeaderActionElements"] {{
    display: none;
}}

/* Titres adaptatifs sous 1200 px, comme dans la version de Streamlit d'origine (évite les titres trop grands sur mobile) */
@media (max-width: 1200px) {{
    [data-testid="stAppViewContainer"] [data-testid="stMain"] h1 {{
        font-size: calc(1.4rem + 1.8vw);
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMain"] h2 {{
        font-size: calc(1.35rem + 1.2vw);
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMain"] h3 {{
        font-size: calc(1.3rem + 0.6vw);
    }}
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown('<div class="titre-accueil" style="text-align:center;width:100%;"><h1 style="color:black;background-color:#f7a900;border:#fc1c24;border-style:solid;border-radius:5px;">TABLEAU DE BORD INTERACTIF DE LA BOITE A IDEES DIGITALE</h1></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;width:100%;"><span style="color:white;background-color:rgba(0,0,0,0.55);border-radius:5px;padding:2px 10px;font-size:0.85rem;">Données fictives — démonstration</span></div>', unsafe_allow_html=True)

#Espacement
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")

# Dictionnaire contenant les informations d'identification des utilisateurs autorisés
# (comptes de démonstration : la direction voit toutes les zones, l'agence ne voit que sa zone)
users_credentials = {
    'direction': {'password': 'demo', 'Zone': TOUTES_LES_ZONES},
    'agence': {'password': 'demo', 'Zone': 'Zone Centre'},
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

# Comptes de démonstration affichés sous le formulaire
if not st.session_state.authenticated:
    st.markdown('<div style="text-align:center;width:100%;"><p style="color:white;background-color:rgba(0,0,0,0.55);border-radius:5px;padding:8px 12px;display:inline-block;margin-top:8px;">Comptes de démonstration : <b>direction</b> / <b>demo</b> (toutes les zones) — <b>agence</b> / <b>demo</b> (Zone Centre)</p></div>', unsafe_allow_html=True)

# ...

if st.session_state.authenticated:
    dashboard_users(st.session_state.Zone)

