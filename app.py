import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO

def read_gd(sharingurl):
    file_id = sharingurl.split('/')[-2]
    download_url='https://drive.google.com/uc?export=download&id=' + file_id
    url = requests.get(download_url).text
    csv_raw = StringIO(url)
    return csv_raw

# --- Load Data from Google Drive ---
url = "https://drive.google.com/file/d/1IG4ssmcRMPbR09HroNQmCXDY-O6Jn84R/view?usp=sharing"
gdd = read_gd(url)
df = pd.read_csv(gdd)

# Clean column names
col_name = [col.strip().lower().replace(' ', '_') for col in df.columns]
df.columns = col_name 

# Rename columns for easier access
df = df.rename(columns={
    'temparature': 'tmp',
    'humidity': 'hum',
    'moisture': 'mst',
    'soil_type': 'sot',
    'nitrogen': 'N',
    'phosphorous': 'P',
    'potassium': 'K',
    'crop_type': 'crp',
    'fertilizer_name': 'fer'
})

df['sot'] = df['sot'].astype('object')

# Encode soil type column
le_sot = LabelEncoder()
df['sot_en'] = le_sot.fit_transform(df['sot'])

# --- Recommendation Function ---
def recommend_crp_fer(tmp, hum, mst, sot, N, P, K, n_exact=1, n=5, tol=1e-2):
    sot_clean = str(sot).strip().lower()  # Clean the input soil type
    
    # Exact match mask with isclose for numeric columns and == for categorical (soil type)
    mask = (
        np.isclose(df['tmp'], tmp, atol=tol) &
        np.isclose(df['hum'], hum, atol=tol) &
        np.isclose(df['mst'], mst, atol=tol) &
        (df['sot'] == sot_clean) &  # Compare raw soil type for exact match
        np.isclose(df['N'], N, atol=tol) &
        np.isclose(df['P'], P, atol=tol) &
        np.isclose(df['K'], K, atol=tol)
    )

    # Find exact matches based on the mask
    exact_matches = df[mask]
    recs = []
    exact_indices = []

    if not exact_matches.empty:
        # For exact matches, append them to recs with True (exact match flag)
        for idx, row in exact_matches.head(n_exact).iterrows():
            crp = row['crp']
            fer = row['fer']
            recs.append((crp, fer, True))  # True indicates exact match
            exact_indices.append(idx)

    # Nearest neighbor calculation (for closest matches)
    try:
        sot_en = le_sot.transform([sot_clean])[0]  # Encode the input soil type
    except ValueError:
        sot_en = df['sot_en'].mode()[0]  # Use the mode if input soil type is unknown
    
    user_vec = np.array([tmp, hum, mst, N, P, K, sot_en])
    num_features = ['tmp', 'hum', 'mst', 'N', 'P', 'K', 'sot_en']
    dataset_vecs = df[num_features].values
    dists = np.linalg.norm(dataset_vecs - user_vec, axis=1)

    # Exclude exact match indices by setting their distances to np.inf
    if exact_indices:
        dists[exact_indices] = np.inf  # Avoid selecting exact matches again

    # Get top n closest matches
    top_indices = np.argsort(dists)[:n]

    # Append the closest matches as False (not exact)
    for idx in top_indices:
        crp = df.iloc[idx]['crp']
        fer = df.iloc[idx]['fer']
        recs.append((crp, fer, False))  # False indicates closest match

    return recs


# --- Streamlit UI ---
st.title("Soil Recommendation App")

# Input fields
tmp = st.number_input('Temperature', value=25.0)
hum = st.number_input('Humidity', value=50.0)
mst = st.number_input('Moisture', value=40.0)
sot = st.selectbox('Soil Type', sorted(df['sot'].unique()))  # Use the unique soil types from the dataset
N = st.number_input('Nitrogen', value=10.0)
P = st.number_input('Phosphorous', value=10.0)
K = st.number_input('Potassium', value=10.0)

if st.button('Get Recommendations'):
    recs = recommend_crp_fer(
        tmp=tmp, hum=hum, mst=mst, sot=sot, N=N, P=P, K=K, n_exact=1, n=5
    )
    for i, (crp, fer, is_exact) in enumerate(recs, 1):
        label = "Exact match" if is_exact else "Closest match"
        st.write(f"**Recommendation {i}:** Crop = `{crp.title()}`, Fertilizer = `{fer.title()}`  _({label})_")