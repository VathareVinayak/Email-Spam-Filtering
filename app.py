import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model and vectorizer
with open('spam_detector_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load the dataset
df = pd.read_csv('mail_data.csv')
mail_data = df.where((pd.notnull(df)), '')
mail_data['Category'] = mail_data['Category'].replace({'spam': 0, 'ham': 1})

# Title
st.title("Spam Detection System")

# Description
st.write("This model predicts whether an email is spam or not.")

# Sidebar with links
with st.sidebar:
    st.link_button("Contact Us", "https://www.linkedin.com/in/vinayak-vathare-4bb135279/")
    st.link_button("Contribute", "https://github.com/VathareVinayak")

    with st.expander("Developer Info"):
        st.write("""
            - [Vinayak Vathare](https://www.linkedin.com/in/vinayak-vathare-4bb135279/)
              - Email: vinayak.vathare2004@gmail.com
        """)
        # Optionally, you can add an image
        # st.image("https://static.streamlit.io/examples/dice.jpg")

# User input for email content
input_mail = st.text_area("Enter the email content here:", height=200)

if st.button("Predict"):
    # Convert text to feature vectors
    input_data_features = vectorizer.transform([input_mail])

    # Make prediction
    prediction = model.predict(input_data_features)

    if prediction[0] == 1:
        st.success('This is a Ham mail.')
    else:
        st.error('This is a Spam mail.')

# Data Visualization
st.subheader("Data Visualization")

# 1. Distribution of Spam vs. Ham emails
st.write("### Distribution of Spam vs. Ham emails")
fig, ax = plt.subplots()
sns.countplot(x='Category', data=mail_data, ax=ax)
ax.set_title('Distribution of Spam vs. Ham emails')
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Spam', 'Ham'])
st.pyplot(fig)

# 2. Length of Emails: Spam vs. Ham
st.write("### Length of Emails: Spam vs. Ham")
mail_data['Message_Length'] = mail_data['Message'].apply(len)
fig, ax = plt.subplots()
sns.histplot(data=mail_data, x='Message_Length', hue='Category', multiple='stack', bins=50, ax=ax)
ax.set_title('Length of Emails: Spam vs. Ham')
ax.set_xlabel('Message Length')
ax.set_ylabel('Count')
ax.legend(['Ham', 'Spam'])
st.pyplot(fig)
