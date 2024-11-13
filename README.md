![Banner](./logo.png)

# Intent Detection in Customer Support Requests using LLMs

A machine learning application for intent detection in customer support requests, helping to streamline and automate support operations by understanding and categorizing user intents.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Overview

This project aims to classify customer support requests into distinct intents, which can help organizations respond to common inquiries more efficiently. The model is built using PyTorch, with a Flask web interface for ease of use.

## Features

- **Machine Learning Model**: A deep learning model for classifying intents in customer support messages.
- **Web Interface**: A Flask-based web interface where users can input support queries to see the detected intent.
- **Pre-trained Checkpoints**: Uses trained model checkpoints for faster inference.

## Technologies Used

- **Python**: Core programming language
- **PyTorch**: Deep learning framework for building and training the intent classification model
- **Flask**: Web framework for building the web interface
- **Jupyter Notebook**: For data analysis and experimentation
- **HTML**: For the web page templates

## Project Structure

```plaintext
intent-detection-project/
│
├── src/
│   ├── app.py              # Flask application
│   ├── model.py            # Model architecture and loading functions
│   ├── data.py             # Data loading and preprocessing
│   ├── setup.py            # Setup configurations
│   ├── train_eval.py       # Model training and evaluation
│   ├── predict.py          # Inference script for making predictions
│
├── templates/
│   └── index.html          # HTML template for Flask web interface
│
├── models/
│   └── model_0.pth         # Placeholder for pre-trained model checkpoint
│
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── intent-detection-in-customer-support.ipynb # Jupyter notebook with data exploration
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Steps

1. Clone the repository

```bash
git clone https://github.com/Abderahmanvt7/intent-detection-in-customer-support-using-llms.git
cd intent-detection-in-customer-support-using-llms
```

2. Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1mDAeiSWxFo_hSf_a4fx3Nggxbx36f0-e/view?usp=drive_link) and place it in the models/ directory. This file will be named model_0.pth by default.

3. Create and activate a virtual environment

```bash
pipenv shell
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

5. Run the application

```bash
python src/app.py
```

## Usage

Open the web application and enter a customer support query.

1. Click on "Detect Intent" to see the predicted intent displayed on the page.
2. You can also directly use the predict.py script to classify an intent

```bash
python src/predict.py --input "Your customer support query here"
```

## Results

The model achieves approximately 93% accuracy on the test dataset, effectively classifying intents in customer support messages with high reliability.

## Limitations and Future Work

### Limitations

- **Limited Dataset:** The current model is trained on a relatively small dataset, which may limit its performance across diverse customer queries. Increasing the dataset size and diversity could potentially improve its accuracy.
- **Intent Overlap:** Some intents have overlapping language patterns, which may make it challenging for the model to differentiate them accurately.
- **Lack of Context Awareness:** The model processes each query independently, which may limit understanding for queries that rely on previous - interactions or contextual information.

### Future Work

- **Model Improvement:** It is possible to enhance accuracy by exploring other deep learning models or hybrid approaches, such as combining rule-based methods with deep learning.
- **Multi-language Support:** Extending the model to support multiple languages could broaden its applicability for international customer support.
- **Contextual Analysis:** Incorporating context-awareness techniques, like memory networks, may improve the model’s performance in cases where understanding context is essential.
- **Interactive Feedback Loop:** Developing a feedback mechanism that allows users to correct misclassified intents could help refine the model over time, with corrections feeding - back into training data to improve accuracy.

## Contributing

Contributions to improve the Arabic Sign Language Detector are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any problems or have suggestions:

1. Check the [Issues](https://github.com/Abderahmanvt7/intent-detection-in-customer-support-using-llms/issues) page
2. Create a new issue if needed
3. Contact the maintainers

---

This project is part of an ongoing effort to make technology more accessible and to bridge communication gaps. We hope it serves as a stepping stone towards more comprehensive sign language interpretation systems.

---

Made with ❤️ by [Abderahman HAMILI](https://abderahmanhamili.vercel.app/)
