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
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix



def statistical_classifier(train_messages: List[str], train_labels: List[str], test_messages: List[str]) -> NDArray[np.str_]:

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

    # Using .fit_transform to to build a vocabulary of all unique words then converts each message into a vector of word counts.
    # As usual i was curious how fit_transform accomplish this, so i did some digging:
        # So fit and transform are separate steps that fit_transform combines into one process.
        # the "fit" portion is what analyzes the training set and builds a unique vocabulary from it.
            # I think of it as its learning the needed parameters to create the text vectorizer lookup table.
        # The "transform" portion uses the learned parameters/vocabulary/lookup table to transform each message into a vector
    # This is basically where the training set is prepped for the model using fit_transform.
    X_train: csr_matrix = vectorizer.fit_transform(train_messages)

    # Initializing a Naive Bayes classifier object an empty model
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
    model: MultinomialNB = MultinomialNB()

    # This is where the actual learning happens
    # Training the model with teh labels and the vectorized data so it can learn the underlying patterns
        # It does this by adjusting its parameters basaed on the input data X_train and the labels train_labels
        # So in theory, the more data you feed it, the better its pattern recognition becomes
        # Which increases is ability to guess the right outcome.
    model.fit(X_train, train_labels)

    # Vectorizing the test set, by converting the test message to the same vector format
    # We only need to do transform here because we need to vectorize the test data
        #using the same list of unique words/covabualry that was generated in the fit_trasform stage above.
        # There is not need to do a fit, because that will now create a new vocabulary
        # This potentially breaks the model and leads to overfitting, because you ar enow testing on the same vectorized data set.
    X_test: csr_matrix = vectorizer.transform(test_messages)

    # Using what the model learned from the training it did earlier "model.fit"
    # to predict if the test message is spam or not
    # I wanted all predictions and not just the first so i removed [0] that was selecting only the first element.
    prediction: NDArray[np.str_] = model.predict(X_test)

    return prediction

    



def main() -> None:
    train_messages: List[str] = [
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

    train_labels: List[str] = [
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

    test_messages: List[str] = [
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

    prediction: NDArray[np.str_] = statistical_classifier(train_messages, train_labels, test_messages)

    # Printing out test messages ane their predictions
    for i in range(len(test_messages)):
        print(f"{test_messages[i]}: {prediction[i]}\n")


if __name__ == "__main__":
    main()


"""
So let's compare my week 1 symbolic Ai spam classifier to this weeks statistical spam classifier.
For my week one classifier i used a symbolic Ai approach with hardcoded rules. Rules like if any of these spam words were detected,
its spam, if the message was above aor below a certain length, its spam. But my statistical spam classifier for this week takes a different approach.
Instead of hard coded rules, used MultinomialNB and Countvectorizer to build and train a model to predict whether a message is spam or not.
Instead of hardcoded rules, i used supervised learning; gave it some trainin data with the lables spam, not spam. This allowed the model to learn the underlying pattern.
This pattern allows it to infer if a message was spam or not.

When it comes to scalability the statistical classifier wins by default for multiple reasons.
If you have a million messages, you would have to manually code the logic to catch spam.
You would have to constantly manually update the rules as the identifiers change.
With the statistical model, if you train it on a large enough dataset, it will be able to recognize spam no matter how the inut scales.
Updating it is as simple as training it on new data.

The statistical model is easier to maintain because you train it on new data if new patterns emerge that it's not picking up.
The symbolic classifier qould require a lot of manual effort that is never ending.

The statistical model would be more accurate with less false positives, becasue it has the ability to handle spam that it has never seen before.
While the symbolic classifier would not know what to do with spam messages that it has no rules for. But when the symbolic model does classify an email as spam, you know why
because the rules are right there. The statistical model on the other hand is like a black box, it decided based on probability.

To wrap this up, the statistical model is definitley better, easier to maintain, more efficient and scales easily too. I am sure there are situatons where the symbolic model would wrok great,
Mayeb in deterministic envronmemts that don't change unless done manually.

"""
