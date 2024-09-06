import streamlit as st
import pickle
import numpy as np
from streamlit_text_rating.st_text_rater import st_text_rater

@st.cache_data
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_country = data['le_country']
le_edLevel = data['le_edLevel']
le_remoteWork = data['le_remoteWork']

def show_predict_page():
    st.title("Developers Salary Prediction")

    st.write("""#### Please provide some information to us in order to predict the salary""")

    countries = ("United States of America",
                "Germany",
                "United Kingdom of Great Britain and Northern Ireland",
                "India",
                "Canada",
                "Brazil",
                "France",
                "Spain",
                "Netherlands",
                "Italy",
                "Australia",
                "Poland")

    education = ("Bachelor's degree", 
                "Master's degree", 
                "Less than a Bachelors",
                "Postgraduate")

    remoteWork = ("Full in-person",
                "Fully remote",
                "Hybrid (some remote, some in-person)")

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education", education)
    remoteWork = st.selectbox("Work Mode", remoteWork)

    code_experience = st.slider("Years of Coding Experience", 0, 50, 5)
    work_experience = st.slider("Years of Working Experience", 0, 40, 1)

    def checkbox_columns(category, items):
        cols = st.columns(2)
        with cols[0]:
            st.write(f"Worked with in PAST year")
            past = {item: st.checkbox(f"{item}", key=f"{item}_past_{category}") for item in items}
        with cols[1]:
            st.write(f"Want to work with NEXT year")
            next = {item: st.checkbox(f"{item}", key=f"{item}_next_{category}") for item in items}
        return past, next

  
    enter = st.button("Predict Salary")
    if enter:
        X = np.array([[country, education, remoteWork, code_experience, work_experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_edLevel.transform(X[:, 1])
        X[:, 2] = le_remoteWork.transform(X[:, 2])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

        st.markdown("Awesome App ?")
        for text in ["Is this helpful?"]:
            response = st_text_rater(text=text)

if __name__ == "__main__":
    show_predict_page()
