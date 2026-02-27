# Symbolic Spam Classifier

**Course:** CSC6313 AI Foundations  
**Week:** 01 Symbolic AI  
**Author:** Shaun Clarke

---

## Project Overview

This project implements a **rule-based spam classifier** using symbolic AI techniques the kind of explicit, human-written reasoning that characterized early AI systems before machine learning existed. No trained models or ML libraries are used. All classification logic is hand-crafted `if/else` rules that encode domain knowledge directly in code.

---

## How It Works

The classifier applies four independent rules to any input string. If any rule is triggered, the message is classified as `"spam"`. If none fire, it returns `"not spam"`.

### Rule 1 Spam Keywords
Checks for the presence of known spam phrases (case-insensitive):
- `"buy now"`, `"free"`, `"winner"`, `"$$$"`

### Rule 2 Excessive Capitalization
Counts the ratio of uppercase letters to total letters. If **40% or more** of the alphabetic characters are uppercase, the message is flagged as spam. This catches messages like `"BUY NOW LIMITED TIME OFFER"` without requiring an exact keyword match.

### Rule 3 Spam Punctuation Patterns
Checks for clusters of repeated punctuation commonly used to grab attention in spam:
- `"!!!"`, `"???"`, `"***"`

### Rule 4 Extreme Message Length
Messages that are **fewer than 15 characters** (too short to be legitimate) or **more than 2000 characters** (unusual bulk content) are flagged as spam.

---

## Example Behavior

```
Input:  "BUY NOW!!! Limited time offer $$$"
Output: spam

Input:  "Can we meet at 3pm tomorrow?"
Output: not spam

Input:  "FREE WINNER!!!"
Output: spam

Input:  "Hey"
Output: spam   ← too short (< 15 characters)
```

---

## Prerequisites

- Python 3.10+
- No external libraries required standard library only

---

## How to Run

```bash
python symbolic_classifier.py
```

The program will prompt you to enter a message, then print `spam` or `not spam`. Type `exit` to quit.

```
So here's the deal ...
You give me some text right , and i will pretend it is an email ...
Then classify it as spam or not, if you get bored type exit ... Sounds good?

Enter your email content here:>
```

---

## Design Notes

**Why symbolic AI?** Symbolic systems encode human expertise as explicit rules they are fully transparent and explainable. Every decision can be traced back to a specific rule, unlike a neural network. The tradeoff is brittleness: rules must be written by hand and may not generalize well to inputs the designer didn't anticipate.

**Capitalization threshold:** The 40% uppercase threshold was chosen to allow normal acronyms and sentence-starting capitals without triggering false positives, while still catching all-caps spam messages.

**Input safety:** The `get_user_input` function handles empty strings with a `ValueError`, keyboard interrupts gracefully, and provides the user a clear exit path none of these are strictly required for the classifier logic but reflect good defensive programming habits.

---

## File Structure

```
week01/
├── symbolic_classifier.py    # Single-file implementation
└── README.md                 # This file
```