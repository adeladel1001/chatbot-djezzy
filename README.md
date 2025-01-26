# Djezzy Chatbot

This project is a chatbot developed for **Djezzy**, using simple deep learning techniques to provide automated responses to user inquiries. The chatbot is built with Python and deep learning frameworks to process and respond to user inputs efficiently.

## Features

- **Automated response system** based on deep learning.
- **Natural Language Processing (NLP)** for improved understanding.
- **User-friendly interface** for seamless interactions.
- **Customizable responses** to fit Djezzy's needs.

## Technologies Used

- **Programming Language:** Python
- **Frameworks & Libraries:**
  - TensorFlow / PyTorch
  - NLTK / spaCy
  - Flask (for serving the chatbot)
- **Frontend:** HTML, CSS, JavaScript

## Installation

To run the project locally, follow these steps:


### 3. Train the Model
Run the following command to train the chatbot model:
```bash
python train_model.py
```

### 4. Start the Chatbot Server
After training, start the Flask server:
```bash
python app.py
```

### 5. Access the Chatbot
Open your web browser and go to:
```
http://localhost:5000
```

## Usage

1. Input a query related to Djezzy services.
2. The chatbot processes the query using deep learning.
3. The response is displayed instantly.

## Folder Structure

```
├── chatbot
├── intents.json
├── chatbot_model.h5
├── train_model.py
├── frontend
│   ├── index.html
│   ├── css/
│   │   ├── styles.css
│   ├── js/
│   │   ├── script.js
├── app.py
└── README.md
```

## Example Interaction

- **User:** "What are the latest offers from Djezzy?"
- **Chatbot:** "Djezzy currently offers 50GB for 2000DA valid for 30 days."

## Potential Improvements

- Enhance the model with more training data for better accuracy.
- Deploy the chatbot to cloud services for scalability.
- Add voice interaction features.

## Contribution

Feel free to contribute by submitting pull requests or reporting issues.



