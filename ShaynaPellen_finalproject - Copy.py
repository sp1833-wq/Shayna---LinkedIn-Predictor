# #!/usr/bin/env python
# # coding: utf-8
#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load & clean data ---
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x): 
    return np.where(x == 1, 1, 0)

ss = pd.DataFrame({
    'sm_li': clean_sm(s['web1h']),
    'income': np.where(s['income'] <= 9, s['income'], np.nan),
    'education': np.where(s['educ2'] <= 8, s['educ2'], np.nan),
    'parent': clean_sm(s['par']),
    'married': clean_sm(s['marital']),
    'female': clean_sm(s['gender']),
    'age': np.where(s['age'] <= 98, s['age'], np.nan)
}).dropna()

X = ss[['income','education','parent','married','female','age']]
y = ss['sm_li']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=842)
model = LogisticRegression(class_weight='balanced', random_state=842).fit(X_train, y_train)

# --- UI ---
# Big title + italic subtitle
st.markdown("<h1 style='text-align: center;'>LinkedIn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>Shayna Pellen Final Project</p>", unsafe_allow_html=True)
st.write("Find Out Who’s Likely to Use LinkedIn")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg", width=50)
st.sidebar.markdown("### What’s your story?")

# Income dropdown
income_options = {
    1: "Less than $10,000",
    2: "10 to under $20,000",
    3: "20 to under $30,000",
    4: "30 to under $40,000",
    5: "40 to under $50,000",
    6: "50 to under $75,000",
    7: "75 to under $100,000",
    8: "100 to under $150,000",
    9: "$150,000 or more"
}
income_label = st.sidebar.selectbox(
    "Income (Household):",
    options=list(income_options.values()),
    help="Income categories are based on household annual income."
)
income = [k for k,v in income_options.items() if v == income_label][0]

# Education dropdown
education_options = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree from a college or university",
    6: "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    7: "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    8: "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"
}
education_label = st.sidebar.selectbox(
    "Education Level:",
    options=list(education_options.values()),
    help="Education categories are based on highest level of school/degree completed."
)
education = [k for k,v in education_options.items() if v == education_label][0]

# Other inputs
parent = st.sidebar.selectbox("Parent?", ["No","Yes"])
married = st.sidebar.selectbox("Married?", ["No","Yes"])
female = st.sidebar.selectbox("Gender", ["Male","Female"])
age = st.sidebar.slider("Age", 18, 98, 30)

# Convert inputs to numeric sample
sample = pd.DataFrame({
    'income':[income],
    'education':[education],
    'parent':[1 if parent=="Yes" else 0],
    'married':[1 if married=="Yes" else 0],
    'female':[1 if female=="Female" else 0],
    'age':[age]
})

# Prediction
prob = model.predict_proba(sample)[0][1]
pred_class = model.predict(sample)[0]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Prediction","Probability Curve","Exploratory Analysis"])

with tab1:
    st.metric("Classification", "LinkedIn User ✅" if pred_class==1 else "Not a LinkedIn User ❌")
    st.metric("Probability", f"{prob:.2%}")

with tab2:
    ages = list(range(18,99))
    samples = pd.DataFrame({
        'income':[income]*len(ages),
        'education':[education]*len(ages),
        'parent':[1 if parent=="Yes" else 0]*len(ages),
        'married':[1 if married=="Yes" else 0]*len(ages),
        'female':[1 if female=="Female" else 0]*len(ages),
        'age':ages
    })
    probs = model.predict_proba(samples)[:,1]
    st.line_chart(pd.DataFrame({"Age":ages,"Probability":probs}).set_index("Age"))

with tab3:
    st.subheader("Exploratory Analysis")

    # Income vs LinkedIn usage
    st.write("**LinkedIn Usage by Income Level**")
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.barplot(x="income", y="sm_li", data=ss, palette="Blues", ci=None, ax=ax1)
    ax1.set_xlabel("Income Level (1-9)")
    ax1.set_ylabel("Proportion LinkedIn Users")
    ax1.set_title("Income vs LinkedIn Usage")
    st.pyplot(fig1)

    # Education vs LinkedIn usage
    st.write("**LinkedIn Usage by Education Level**")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.barplot(x="education", y="sm_li", data=ss, palette="Greens", ci=None, ax=ax2)
    ax2.set_xlabel("Education Level (1-8)")
    ax2.set_ylabel("Proportion LinkedIn Users")
    ax2.set_title("Education vs LinkedIn Usage")
    st.pyplot(fig2)

    # Simplified correlation bar chart
    st.write("**Feature Correlation with LinkedIn Usage**")
    corr_target = ss[['income','education','parent','married','female','age','sm_li']].corr()['sm_li'].drop('sm_li')
    fig3, ax3 = plt.subplots(figsize=(8,4))
    corr_target.plot(kind='bar', color="steelblue", ax=ax3)
    ax3.set_ylabel("Correlation with LinkedIn Usage")
    ax3.set_title("Feature Importance (Correlation)")
    st.pyplot(fig3)
st.markdown(
    """
    <style>
    /* Rounded corners for widgets */
    .stButton>button, .stSelectbox, .stSlider {
        border-radius: 10px;
    }

    /* Custom success/warning/error colors */
    .stMetric {
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 8px;
    }
    .stMetric label {
        color: #0077B5; /* LinkedIn blue for labels */
    }

    /* Highlight hover states */
    .stButton>button:hover {
        background-color: #60A5FA !important;
        color: white !important;
    }

    /* Adjust line spacing */
    .css-1aumxhk { line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True
)

# # # Final Project
# # ## Shayna Pellen
# # ### Date 12/9

# # ***

# # #### Q1

# # Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# # In[101]:


# import pandas as pd

# s = pd.read_csv("social_media_usage.csv")


# # In[102]:


# s.shape


# # ***

# # #### Q2

# # Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works 

# # In[103]:


# import pandas as pd
# import numpy as np

# def clean_sm(x):  
#     return np.where(x == 1, 1, 0)

# #Dataframe
# df = pd.DataFrame({
#     'col1': [1, 2, 1],
#     'col2': [2, 0, 1]
# })    

# #test function (will print the array output)
# print(clean_sm(df))

# #apply function to dataframe (will print the dataframe output)
# df_cleaned = df.apply(clean_sm)
# print(df_cleaned)


# # Create a function that takes a number as input and returns whether the person is a LinkedIn user. If the input equals 1, the function should return 1 (LinkedIn user), otherwise it will show a 0 (not a LinkedIn user).

# # ***

# # #### Q3

# # Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# # In[104]:


# ss = pd.DataFrame({
#     'sm_li': clean_sm(s['web1h']),  # LinkedIn usage column in survey
#     'income': np.where(s['income'] <= 9, s['income'], np.nan),
#     'education': np.where(s['educ2'] <= 8, s['educ2'], np.nan),
#     'parent': clean_sm(s['par']),
#     'married': clean_sm(s['marital']),
#     'female': clean_sm(s['gender']),
#     'age': np.where(s['age'] <= 98, s['age'], np.nan)
# })

# ss = ss.dropna()
# ss.head()


# # In[105]:


# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.countplot(x='sm_li', data=ss, palette='coolwarm')
# plt.title("LinkedIn Users vs Non-Users")
# plt.show()


# # In[106]:


# # Parent vs LinkedIn usage
# sns.countplot(x='parent', hue='sm_li', data=ss, palette='coolwarm')
# plt.title("Parent Status vs LinkedIn Usage")
# plt.show()

# # Married vs LinkedIn usage
# sns.countplot(x='married', hue='sm_li', data=ss, palette='coolwarm')
# plt.title("Married Status vs LinkedIn Usage")
# plt.show()

# # Female vs LinkedIn usage
# sns.countplot(x='female', hue='sm_li', data=ss, palette='coolwarm')
# plt.title("Gender vs LinkedIn Usage")
# plt.show()

# # Income vs LinkedIn usage
# sns.countplot(x='income', hue='sm_li', data=ss, palette='coolwarm')
# plt.title("Income vs LinkedIn Usage")
# plt.show()

# # Education vs LinkedIn usage
# sns.countplot(x='education', hue='sm_li', data=ss, palette='coolwarm')
# plt.title("Education vs LinkedIn Usage")
# plt.show()


# # ***

# # #### Q4

# # Create a target vector (y) and feature set (X)

# # In[107]:


# y = ss['sm_li']
# print(y)

# X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]
# print(X)


# # Splitting the data into two sets linkedin user that you want to predict is the y and the features you want to use to make the prediction is the x.

# # ***

# # #### Q5

# # Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# # In[108]:


# from sklearn.model_selection import train_test_split


# # In[123]:


# y = ss['sm_li']

# X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=842,
# )


# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)


# # ***

# # #### Q6

# # Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# # In[124]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split


# # In[125]:


# logit_model = LogisticRegression(class_weight='balanced', random_state=842)

# # Fit the model on training data
# logit_model.fit(X_train, y_train)


# # ***

# # #### Q7 

# # Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# # In[126]:


# from sklearn.metrics import accuracy_score, confusion_matrix
# y_pred = logit_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Model Accuracy:", accuracy)

# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm)


# # explain confusion matrix here and then model accuracy - how much of the time is it arrucate

# # ***

# # #### Q8

# # Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# # In[127]:


# import pandas as pd
# from sklearn.metrics import confusion_matrix

# # Force confusion matrix to include both classes (0 and 1)
# cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# # Create DataFrame with informative labels
# cm_df = pd.DataFrame(
#     cm,
#     index=['Actual 0 (Non-user)', 'Actual 1 (User)'],
#     columns=['Predicted 0 (Non-user)', 'Predicted 1 (User)']
# )

# print("Confusion Matrix as DataFrame:")
# print(cm_df)


# # ***

# # #### Q9

# # Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# # write how you calculated it by hand

# # In[128]:


# from sklearn.metrics import classification_report, confusion_matrix

# # Example confusion matrix
# cm = confusion_matrix(y_test, y_pred, labels=[0,1])
# TN, FP, FN, TP = cm.ravel()

# # Manual calculations
# precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# print("Precision (class 1):", precision)
# print("Recall (class 1):", recall)
# print("F1 Score (class 1):", f1)

# # Classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred, labels=[0,1]))


# # ***

# # #### Q10

# # Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# # In[129]:


# import pandas as pd

# # Create new samples
# new_samples = pd.DataFrame({
#     'income': [8, 8],
#     'education': [7, 7],
#     'parent': [0, 0],
#     'married': [1, 1],
#     'female': [1, 1],
#     'age': [42, 82]   # different ages
# })

# # Predict probability of LinkedIn usage (class 1)
# probs = logit_model.predict_proba(new_samples)[:, 1]

# print("Probability Person A (age 42):", probs[0])
# print("Probability Person B (age 82):", probs[1])


# # In[130]:


# import matplotlib.pyplot as plt

# # Ages and predicted probabilities
# ages = [42, 82]
# probs = logit_model.predict_proba(new_samples)[:, 1]

# # Plot the points
# plt.plot(ages, probs, marker='o', linestyle='-', color='blue')

# # Annotate each point with its probability
# for age, prob in zip(ages, probs):
#     plt.text(age, prob + 0.02, f"{prob:.2f}", ha='center')

# # Labels and title
# plt.xlabel("Age")
# plt.ylabel("Predicted Probability of LinkedIn Usage")
# plt.title("Predicted Probability of LinkedIn Usage by Age")
# plt.grid(True)

# # Show the plot
# plt.show()


# # In[ ]:





# # ***
