"""
Name: Shaun Clarke
Course: CSC6313 Ai Foundations
Instructor: Margaret Mulhall
Module: 2
Assignment: Statistical Spam Classifier

In this assignment, you will analyze the below Statistical Spam Classifier, and compare to your week 1 assignment (Symbolic Spam Classifier). 
"""

# Mental note, a document is one piece of text input that is treated as a single unit.
# example:
    # one entire text message would be one document
    # one entire email would be one document
    # one entire slack message would be one document.

# The CountVectorizer is being imported from the scikit-learn library
from sklearn.feature_extraction.text import CountVectorizer
# Importing MultinomialNB, this is the Naive Bayes algortihm that calculates probabilities. This is where the magic happens
from sklearn.naive_bayes import MultinomialNB
from typing import List


def statistical_classifier(train_messages: List, train_labels: List, test_message: List):

    # initializing a word counter object that will convert text to numbers
    # The CountVectorizer from the scikit-learn library.
    # The CountVectorizer is important because it converts the inout text into numerical feature vectors based on the frequency count.
    # How does it convert the input text into numerical feature vectors?
    # Starts by splitting the input text into words, this is called tokenization.
    # It then counts the occurrence of each word.
    # From that count it builds a unique vocabulary of words from across all documents and each word has an index column.
    # Using the vocabulary, each document is then represented as a vector of words based on the words frequency count and this is pretty interesting.
    # Lets say when the vocabulary was built it has ["buy","now","free"]
        # Then we use the vocabulary to vectorize the document "buy now".
        # Each word in the document appeared once so the "buy now" document as a vector would be [1, 1, 0]
        # Here's the interesting part, why is there a 0? This is because vector positions correspond to every word in the vocabulary, not document.
        # So the 0 is because free is not in the document "buy now"
    vectorizer: CountVectorizer = CountVectorizer()

    # Using .fit_transform to to build a list of all unique words then converts each message into a vector of word counts.
    # This is basically the training set
    X_train = vectorizer.fit_transform(train_messages)

    # Insantiating a Naive Bayes classifier object an empty model
    # This is another important piece because it implements the Multinomial Naive Bayes algo that's typically used in text classification.
    # It is good for text classification because it does really well at handling discrete features/inputs like word counts.
    # Curious why Multinomial Naive Bayes, i did some digging.
        # The word Naive eludes to the method assuming that all the input\features(words in our case) are independent from each other.
        # The word Bayes means it's using the bayes theorem to calculate probabilities
        # Multinomial refers to the frequency of a word occuring, basically basically how many times has this word appeared.
    # So how exactly does it work?
        # So during training the model looks at the frequency of a word appearing in messages from different categories i.e.(spam or not spam) then calculates the probability for each word.
            # So if we have "buy now" showing up a lot in messages classified as spam, this will used to help predict whether a new message is spam or not.
        # During prediction it uses those probabilities to calculate the probability of a message being spam or not.
            # 

    model: MultinomialNB = MultinomialNB()

    # This is where the actual learning happens
    # Training the model with teh labels and the vectorized data
    model.fit(X_train, train_labels)

    # Vectorizing the test set, by converting the test message to the same vector format
    # using the same list of unique words as above
    # As we stated earlier, we cant do math on words, vectorize it to the specs we vectorized the training set with
    X_test = vectorizer.transform(test_message)

    # Using what the model learned from the training it did earlier "model.fit"
    # to predict if the test message is spam or not
    # I wanted all predictions and not just the first so i removed [0] that was selecting only the first element.
    prediction = model.predict(X_test)

    # Printing out test messages ane their predictions
    for i in range(len(test_message)):
        print(f"{test_message[i]}: {prediction[i]}")



def main():
    train_messages: List = [
        "Congratulations! You've won a $1,000 Amazon gift card. Click here to claim your prize now.",
        "URGENT: Your account has been suspended. Verify your information immediately to restore access.",
        "Limited time offer!!! Get cheap meds without prescription. Order today.",
        "You have been selected for a cash reward. Reply YES to receive your funds.",
        "Final notice: Your loan application is approved. Act now to avoid cancellation.",
        "Earn $500 a day working from home. No experience required. Sign up now.",
        "Your package could not be delivered. Confirm your address at the link below.",
        "Hot singles in your area are waiting to chat. Click here to start now.",
        "Bank alert: Suspicious activity detected. Login immediately to secure your account.",
        "Free crypto giveaway! Send 1 ETH and receive 2 ETH instantly.",
        "IRS warning: Unpaid taxes detected. Immediate payment required to avoid arrest.",
        "You've been pre-approved for a credit increase. Click to accept.",
        "Exclusive deal just for you! 90% off luxury watches today only.",
        "Your Apple ID has been locked. Reset your password now.",
        "Claim your inheritance funds before the deadline expires.",
        "Hey, are we still on for dinner tonight at 7?",
        "Your Amazon order has shipped and will arrive tomorrow.",
        "Reminder: Your dentist appointment is scheduled for Monday at 10am.",
        "Please review the attached document and let me know if you have questions.",
        "Can you send me the meeting notes from yesterday?",
        "Your package has been delivered to the front door.",
        "Let's reschedule our call to later this afternoon.",
        "Thanks for your payment. Your receipt is attached.",
        "Happy birthday! Hope you have an amazing day.",
        "Your subscription will renew on February 1st.",
        "I pushed the latest changes to GitHub. Please pull when you get a chance.",
        "Don't forget to submit your assignment before midnight.",
        "Are you free this weekend to help me move?",
        "Your internet service has been restored.",
        "Lunch was great today — we should do that again soon."
    ]

    train_labels: List = [
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam"
    ]

    test_message: List[tuple[str,int]] = [
        "IRS warning: Unpaid taxes detected. Immediate payment required to avoid arrest.",
        "You've been pre-approved for a credit increase. Click to accept.",
        "Exclusive deal just for you! 90% off luxury watches today only.",
        "Your Apple ID has been locked. Reset your password now.",
        "Claim your inheritance funds before the deadline expires.",
        "Hey, are we still on for dinner tonight at 7?",
        "Your Amazon order has shipped and will arrive tomorrow.",
        "Reminder: Your dentist appointment is scheduled for Monday at am.",
        "Please review the attached document and let me know if you have questions.",
        "Can you send me the meeting notes from yesterday?"
    ]

    statistical_classifier(train_messages, train_labels, test_message)


if __name__ == "__main__":
    main()
