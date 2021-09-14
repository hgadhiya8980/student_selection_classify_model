from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Admission.csv")

df_value = df["Decision"].value_counts().keys()

for num, var in enumerate(df_value):
    print(num)
    df["Decision"].replace(var,num, inplace=True)

X = df.drop("Decision", axis=1)
y = df["Decision"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.20, random_state=123)

sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("student_selection_classifie.pkl")

def student_selection_classife(model, GPA, GMAT):
#     for num,var in enumerate(df_value):
#         if var == Decision:
#             Decision = num        
            
    x = np.zeros(len(X.columns))
    x[0] = GPA
    x[1] = GMAT
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    GPA = request.form["GPA"]
    GMAT = request.form["GMAT"]
    
    predicated_price =student_selection_classife(model,GPA, GMAT)

    if predicated_price==1:
        return render_template("index.html", prediction_text="you will be selected")
    elif predicated_price==2:
        return render_template("index.html", prediction_text="you not eligible for this selection process")
    else:
        return render_template("index.html", prediction_text="your performance will great sometimes you will select.")

if __name__ == "__main__":
    app.run()   
    