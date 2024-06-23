# phishing_url_detector
Phishing URL Detector Using ANN

# Phishing URL Detection using ANN

## Overview
This project aims to detect phishing URLs using the PhiUSIIL Phishing URL Dataset and an Artificial Neural Network (ANN) for classification. The dataset consists of 235,795 instances with 56 columns, including various features extracted from URLs.

## Dataset
The dataset used for this project contains the following columns:

1. `FILENAME`: Filename of the URL
2. `URL`: The URL itself
3. `URLLength`: Length of the URL
4. `Domain`: Domain of the URL
5. `DomainLength`: Length of the domain
6. `IsDomainIP`: Whether the domain is an IP address (1 = Yes, 0 = No)
7. `TLD`: Top Level Domain
8. `URLSimilarityIndex`: Similarity index of the URL
9. `CharContinuationRate`: Character continuation rate in the URL
10. `TLDLegitimateProb`: Probability of the TLD being legitimate
11. `URLCharProb`: Probability of characters in the URL being legitimate
12. `TLDLength`: Length of the TLD
13. `NoOfSubDomain`: Number of subdomains
14. `HasObfuscation`: Whether the URL has obfuscation (1 = Yes, 0 = No)
15. `NoOfObfuscatedChar`: Number of obfuscated characters in the URL
16. `ObfuscationRatio`: Ratio of obfuscated characters in the URL
17. `NoOfLettersInURL`: Number of letters in the URL
18. `LetterRatioInURL`: Ratio of letters in the URL
19. `NoOfDegitsInURL`: Number of digits in the URL
20. `DegitRatioInURL`: Ratio of digits in the URL
21. `NoOfEqualsInURL`: Number of '=' characters in the URL
22. `NoOfQMarkInURL`: Number of '?' characters in the URL
23. `NoOfAmpersandInURL`: Number of '&' characters in the URL
24. `NoOfOtherSpecialCharsInURL`: Number of other special characters in the URL
25. `SpacialCharRatioInURL`: Ratio of special characters in the URL
26. `IsHTTPS`: Whether the URL uses HTTPS (1 = Yes, 0 = No)
27. `LineOfCode`: Number of lines of code in the page
28. `LargestLineLength`: Length of the largest line of code
29. `HasTitle`: Whether the page has a title (1 = Yes, 0 = No)
30. `Title`: Title of the page
31. `DomainTitleMatchScore`: Match score between the domain and title
32. `URLTitleMatchScore`: Match score between the URL and title
33. `HasFavicon`: Whether the page has a favicon (1 = Yes, 0 = No)
34. `Robots`: Whether the page has a robots.txt file (1 = Yes, 0 = No)
35. `IsResponsive`: Whether the page is responsive (1 = Yes, 0 = No)
36. `NoOfURLRedirect`: Number of URL redirects
37. `NoOfSelfRedirect`: Number of self redirects
38. `HasDescription`: Whether the page has a description (1 = Yes, 0 = No)
39. `NoOfPopup`: Number of pop-ups
40. `NoOfiFrame`: Number of iframes
41. `HasExternalFormSubmit`: Whether the page has external form submission (1 = Yes, 0 = No)
42. `HasSocialNet`: Whether the page links to social networks (1 = Yes, 0 = No)
43. `HasSubmitButton`: Whether the page has a submit button (1 = Yes, 0 = No)
44. `HasHiddenFields`: Whether the page has hidden fields (1 = Yes, 0 = No)
45. `HasPasswordField`: Whether the page has a password field (1 = Yes, 0 = No)
46. `Bank`: Whether the page is related to banking (1 = Yes, 0 = No)
47. `Pay`: Whether the page is related to payment (1 = Yes, 0 = No)
48. `Crypto`: Whether the page is related to cryptocurrency (1 = Yes, 0 = No)
49. `HasCopyrightInfo`: Whether the page has copyright information (1 = Yes, 0 = No)
50. `NoOfImage`: Number of images on the page
51. `NoOfCSS`: Number of CSS files on the page
52. `NoOfJS`: Number of JavaScript files on the page
53. `NoOfSelfRef`: Number of self-references
54. `NoOfEmptyRef`: Number of empty references
55. `NoOfExternalRef`: Number of external references
56. `label`: Target label (1 = Legitimate, 0 = Phishing)

## Data Preprocessing
1. **Basic Analysis**:
   - Checked the shape, descriptions, and basic information using `df.info()` and `df.describe()`.
   - Verified null values using `df.isnull().sum()`.

2. **Column Dropping**:
   - Dropped the columns `['FILENAME', 'URL', 'Domain', 'Title']` as they are not relevant for prediction.

3. **Label Distribution**:
   - Verified the count of each label category:
     ```plaintext
     1    134850
     0    100945
     Name: count, dtype: int64
     ```

4. **Correlation Analysis**:
   - Found the correlation of the `label` column with other features:
     ```python
     corr_dict = dict(df.corr(numeric_only=True)["label"].sort_values())
     ```

5. **Encoding Categorical Columns**:
   - Used `LabelEncoder` from `sklearn` to encode the `TLD` column:
     ```python
     from sklearn.preprocessing import LabelEncoder
     encoder = LabelEncoder()
     df["TLD"] = encoder.fit_transform(df["TLD"])
     ```

## Model Training
1. **Train-Test Split**:
   - Used `train_test_split` from `sklearn` to split the data into training and testing sets:
     ```python
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import MinMaxScaler

     X = df.drop('label', axis=1)
     y = df['label']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)

     scaler = MinMaxScaler()
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)
     ```

2. **Model Definition**:
   - Defined the ANN model using `Sequential` from `tf.keras.models` and added layers using `Dense`, `Dropout`, and `InputLayer` from `tf.keras.layers`:
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Dense, Dropout, InputLayer

     model = Sequential()
     model.add(InputLayer(shape=(51,)))
     model.add(Dense(1024, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(512, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(256, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(128, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(32, activation="relu"))
     model.add(Dropout(0.3))
     model.add(Dense(1, activation="sigmoid"))

     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
     ```

3. **Model Training**:
   - Trained the model with 5 epochs:
     ```python
     model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=5)
     ```

## Model Evaluation
Evaluated the model using accuracy and loss metrics:
- **Accuracy**: 0.9995
- **Loss**: 0.0020

Generated a classification report and confusion matrix:

- **Classification Report:**
```plaintext
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     33312
           1       1.00      1.00      1.00     44501

    accuracy                           1.00     77813
   macro avg       1.00      1.00      1.00     77813
weighted avg       1.00      1.00      1.00     77813
```

- **Confusion Matrix:**
```plaintext
[[33265    47]
 [    0 44501]]
```

 ## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/riitk/phishing_url_detector.git
    cd phishing_url_detector
    ```

## Acknowledgements
- The dataset was sourced from Kaggle: [Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).
