# Tweet Text Classification with scikit-learn

Αυτό το repository περιέχει ένα Jupyter Notebook (`sentiment.ipynb`) για ταξινόμηση κειμένου (tweets) σε κατηγορίες, χρησιμοποιώντας κλασικά μοντέλα machine learning (Naive Bayes, Logistic Regression, Linear SVM) και τεχνικές όπως TF–IDF και SMOTE για αντιμετώπιση class imbalance.

## Περιγραφή

Στο notebook γίνονται τα ακόλουθα βήματα:

1. **Φόρτωση δεδομένων**
   - Φόρτωση ενός αρχείου `train.csv` με στήλες τύπου:
     - `tweet`: το κείμενο του tweet
     - `label`: η κατηγορία/ετικέτα (π.χ. 0/1)

2. **Προεπεξεργασία κειμένου**
   - Μετατροπή σε lowercase
   - Αφαίρεση mentions (`@user`)
   - Αφαίρεση URLs
   - Καθαρισμός από punctuation & ειδικούς χαρακτήρες
   - Αφαίρεση stopwords με χρήση `nltk`

3. **Vectorization**
   - Μετατροπή των κειμένων σε αριθμητικά features με **TF–IDF** (`TfidfVectorizer`), με χρήση n-grams (1–2).

4. **Διαχωρισμός σε train/validation set**
   - Χρήση `train_test_split` από `sklearn.model_selection`.

5. **Μοντέλα που εκπαιδεύονται**
   - **Model 1: Naive Bayes**
     - `MultinomialNB` πάνω στα TF–IDF features.
   - **Model 2: Logistic Regression (balanced)**
     - `LogisticRegression` με `class_weight='balanced'`.
   - **Model 3: Linear SVC + SMOTE**
     - Oversampling με `SMOTE` για τοtrain set.
     - Εκπαίδευση `LinearSVC`.
   - **Model 4: Linear SVC + Calibrated Probabilities + Threshold**
     - `LinearSVC` τυλιγμένο με `CalibratedClassifierCV` για παραγωγή probabilities.
     - Πειραματισμός με decision threshold (0.5 κ.λπ.) για βελτίωση του F1.

6. **Αξιολόγηση**
   - `classification_report`
   - Confusion matrix
   - `f1_score` (για κλάση 1 και macro)
   - `accuracy_score`
   - Δημιουργία summary table με τα αποτελέσματα όλων των μοντέλων.
