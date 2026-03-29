from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    "Win money now",
    "Hello how are you",
    "Free offer just click",
    "Let's meet tomorrow",
    "Congratulations you won a prize",
    "Are you coming to class?"
]

labels = ["spam", "ham", "spam", "ham", "spam", "ham"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)


model = MultinomialNB()
model.fit(X, labels)

user_input = input("Enter a message: ")
user_data = vectorizer.transform([user_input])

prediction = model.predict(user_data)

print("This message is:", prediction[0])