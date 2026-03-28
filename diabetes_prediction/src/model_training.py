from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(df):
   
    df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 1 if x > 0 else 0)


    y = df["Diabetes_binary"]
    X = df.drop(["Diabetes_012", "Diabetes_binary"], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    return model