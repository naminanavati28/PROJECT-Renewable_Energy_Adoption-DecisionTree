from sklearn.tree import DecisionTreeClassifier
import pickle

def train(df):
    X = df.drop("Adoption", axis=1)
    y = df["Adoption"]
    model = DecisionTreeClassifier()
    model.fit(X, y)
    pickle.dump(model, open("model.pkl", "wb"))
