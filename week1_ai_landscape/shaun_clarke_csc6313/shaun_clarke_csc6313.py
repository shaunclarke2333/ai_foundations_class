"""
Name: Shaun Clarke
Course: CSC6313 Ai Foundations
Instructor: Margaret Mulhall
Assignment: Symbolic Spam Classifier

In this assignment, you will build a simple rule-based spam detector using symbolic AI techniques.
This mirrors early AI systems that relied on human-written rules rather than learning from data.
"""

# This function acts as a simple rule based spam detector
def symbolic_classifier(text_input: str) -> str:
    """
    This symbolic_classifier function takes a body of text.<br>
    Them determines if it's spam or not spam.
    
    :param text_input: This will be the body of text that the user will be promted to enter
    :type text_input: str
    :return: This function returns one of two strings "spam" or "not spam"
    :rtype: str
    """

    # Defining validation sets for spam keywords and puntuations to check for
    spam_keywords: set = {"buy now", "free", "winner", "$$$"}
    spam_punctuation: set = {"!!!", "???", "***"}
    # Making text lowercase
    lower_input: str = text_input.lower()

    # Checking for spam keywords
    for spam_key in spam_keywords:
        if spam_key in lower_input:
            return "spam"
        
    # Checking for excessive capitalization
    percentage: float = 40 # if the amount of uppercase letters is more than this its spam
    uppercase_count: int = 0 # uppercase letter count
    letter_count: int = 0 # counting all letters

    # looping theough input and counting the number of uppercase letters compared to the total number of letters
    for character in text_input:
        # if this character is a letter count it
        if character.isalpha():
            letter_count += 1
            # if this letter is uppercase count it
            if character.isupper():
                uppercase_count += 1
    
    # Making sure we don't crash if letter count is 0.
    if letter_count > 0:
        # Calculating the percentage of letters that are uppercase
        uppercase_letter_percentage: float = round((uppercase_count / letter_count) * 100, 0)
        # If the percetage of uppercase letters are greater than or equal to the percentage variable, its spam
        if uppercase_letter_percentage >= percentage:
            return "spam"

    # Checking for excessiv epunctuations
    for spam_punc in spam_punctuation:
        if spam_punc in lower_input:
            return "spam"
        
    # If input is less than 15 characters or more that 2000 characters, its spam.
    if len(text_input) < 15 or len(text_input) > 2000:
        return "spam"
        


    return "not spam"

# This function handles user inputs
def get_user_input() -> str:
    
    # While loop to make sure the user enters the correct input before it moves on
    while True:
        try:
            # Asking user for input
            get_input = input(
                """
                So here's the deal ...
                You give me some text right , and i will pretend it is an email ...
                Then classify it as spam or not, if you get bored type exit ... Sounds good?\n
                Enter your email content here:>\n
                """
            )

            # Allowing the user to exit the program
            if get_input == "exit":
                print(f"Bye for now ..")
                exit()

            # If input is empty
            if get_input == "":
                raise ValueError
            
            return get_input
            
        # Gracefully handling keyboard interrupt
        except KeyboardInterrupt:
            print(f"\nReally dude, keyboard intterupt? Ok bye.\n")
            exit()
        except ValueError:
            print(f"You missunderstood the mission objective, please try again ...\n")



def main():
   
   # Getting user input
   user_input = get_user_input()
   # Checking input if spam or not
   validation_results = symbolic_classifier(user_input)
   
   print(validation_results)
   
if __name__ == "__main__":
    main()
