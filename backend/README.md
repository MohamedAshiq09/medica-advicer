# Medical Symptoms Checker - Backend

A comprehensive AI-powered medical triage system that uses machine learning algorithms and association rules to provide symptom-based recommendations.

## 🏗️ Architecture

The backend follows a **multi-agent architecture** where each agent has specific responsibilities:

### 🤖 Agents Overview

1. **Preprocessing Agent** (`src/preprocessing.py`)
   - Cleans and normalizes user input text
   - Extracts symptoms using keyword matching
   - Creates structured feature vectors for ML models
   - Handles severity indicators and demographic data

2. **Model Inference Agent** (`src/model_inference.py`)
   - Loads and manages trained ML models
   - Makes triage predictions (self-care/see_doctor/emergency)
   - Supports both text-based (TF-IDF) and feature-based models
   - Provides confidence scores and probability distributions

3. **Apriori Rules Agent** (`src/rules_apriori.py`)
   - Applies association rules mined from symptom patterns
   - Enhances predictions with pattern-based logic
   - Uses both mined rules and medical domain knowledge
   - Provides symptom association insights

4. **Safety Layer Agent** (`src/safety_layer.py`)
   - Applies critical safety overrides for red-flag symptoms
   - Ensures no dangerous recommendations (e.g., self-care for chest pain)
   - Implements hard-coded medical safety rules
   - Generates appropriate medical disclaimers

5. **LLM Layer Agent** (`src/llm_layer.py`)
   - Generates user-friendly explanations (optional)
   - Creates follow-up questions for symptom clarification
   - Uses OpenAI API with strict medical safety constraints
   - Falls back to template-based explanations

## 🧠 Machine Learning Algorithms

### Core Algorithms Used:

1. **Logistic Regression**
   - Primary classifier for triage decisions
   - Excellent interpretability for medical applications
   - Handles both text (TF-IDF) and structured features

2. **Random Forest**
   - Handles non-linear symptom interactions
   - Robust to noisy features
   - Provides feature importance rankings

3. **TF-IDF Vectorization**
   - Converts symptom descriptions to numerical features
   - Captures important medical terminology
   - Handles variable-length text input

4. **Apriori Algorithm**
   - Mines frequent symptom combinations
   - Discovers hidden patterns in medical data
   - Generates association rules for symptom co-occurrence

### Why These Algorithms?

- **Medical Interpretability**: Linear models allow doctors to understand decision factors
- **Data Efficiency**: Work well with limited medical datasets
- **Safety**: Predictable behavior compared to deep learning models
- **Proven**: Widely used in clinical decision support systems

## 📁 Project Structure

```
backend/
├── src/                          # Source code (agents)
│   ├── __init__.py
│   ├── config.py                 # Configuration and constants
│   ├── preprocessing.py          # Preprocessing Agent
│   ├── model_inference.py        # Model Inference Agent
│   ├── rules_apriori.py         # Apriori Rules Agent
│   ├── safety_layer.py          # Safety Layer Agent
│   └── llm_layer.py             # LLM Layer Agent
├── data/                         # Data files
│   ├── sample_symptom_cases.csv  # Sample training data
│   ├── processed_symptom_cases.csv # Processed features
│   └── association_rules.json    # Mined association rules
├── models/                       # Trained models
│   ├── triage_model.joblib      # Main ML model
│   ├── tfidf_vectorizer.joblib  # Text vectorizer
│   └── label_encoder.joblib     # Label encoder
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_train_models.ipynb
│   └── 03_apriori_mining.ipynb
├── main.py                      # FastAPI application
├── run_server.py               # Server startup script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train Models (Optional)

Run the Jupyter notebooks to train models on your data:

```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. 01_eda_preprocessing.ipynb
# 2. 02_train_models.ipynb  
# 3. 03_apriori_mining.ipynb
```

### 3. Start the Server

```bash
# Using the startup script
python run_server.py

# Or directly with uvicorn
uvicorn main:app --reload --port 8000
```

### 4. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🔧 API Endpoints

### Main Endpoints

- `POST /check_symptoms` - Main symptom checking endpoint
- `GET /health` - Health check and system status
- `GET /` - API information

### Testing Endpoints

- `GET /symptoms/vocabulary` - Get symptom vocabulary
- `GET /model/status` - Get model status information
- `POST /test/preprocessing` - Test preprocessing functionality

### Example Request

```json
POST /check_symptoms
{
  "text": "I have severe chest pain and difficulty breathing",
  "age": 45,
  "gender": "M"
}
```

### Example Response

```json
{
  "triage_level": "emergency",
  "confidence": 0.95,
  "explanation": "Your symptoms indicate you should seek immediate medical attention...",
  "detected_symptoms": ["chest_pain", "shortness_breath"],
  "probabilities": {
    "self-care": 0.02,
    "see_doctor": 0.03,
    "emergency": 0.95
  },
  "applied_overrides": [
    {
      "type": "emergency_override",
      "reason": "Critical symptoms detected requiring immediate attention"
    }
  ],
  "applied_rules": [...],
  "follow_up_questions": [...],
  "timestamp": "2024-01-15T10:30:00",
  "model_info": {...}
}
```

## 🛡️ Safety Features

### Multi-Layer Safety System

1. **Red Flag Detection**: Automatically detects emergency symptoms
2. **Safety Overrides**: Hard-coded rules for critical conditions
3. **Conservative Defaults**: When uncertain, recommends medical consultation
4. **Medical Disclaimers**: Always includes appropriate disclaimers
5. **No Self-Diagnosis**: System provides triage, not diagnosis

### Red Flag Symptoms

- Severe chest pain
- Difficulty breathing
- Sudden severe headache
- Loss of consciousness
- Severe bleeding
- Signs of stroke
- Severe allergic reactions

## 🔬 Model Training

### Data Requirements

The system expects training data with these columns:
- `complaint_text`: User's symptom description
- `age`: User's age (optional)
- `gender`: User's gender (optional)
- `triage_label`: Target label (0=self-care, 1=see_doctor, 2=emergency)

### Training Process

1. **Data Preprocessing**: Clean text, extract symptoms, create features
2. **Model Training**: Train multiple algorithms, select best performer
3. **Rule Mining**: Use Apriori to discover symptom patterns
4. **Validation**: Test safety rules and edge cases
5. **Model Saving**: Save trained models and metadata

## 🌐 Integration with Frontend

The backend is designed to work with a Next.js frontend:

- **CORS enabled** for localhost:3000
- **RESTful API** with JSON responses
- **Comprehensive error handling**
- **Detailed response metadata**

## 📊 Monitoring and Logging

- **Health checks** for system status
- **Request logging** for debugging
- **Model performance** tracking
- **Safety override** monitoring

## 🔧 Configuration

Key settings in `src/config.py`:

- **Symptom vocabulary**: Customizable symptom keywords
- **Safety thresholds**: Confidence and override settings
- **Model paths**: File locations for trained models
- **API settings**: CORS, timeouts, etc.

## 🚨 Important Notes

### Medical Disclaimer

⚠️ **This system is for educational and research purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.**

### Limitations

- **Not a replacement** for professional medical advice
- **Limited training data** may affect accuracy
- **Rule-based safety** may not cover all edge cases
- **Requires regular updates** with new medical knowledge

## 🤝 Contributing

1. Follow the agent-based architecture
2. Add comprehensive tests for new features
3. Update safety rules when adding new symptoms
4. Document all medical assumptions and sources
5. Test thoroughly with edge cases

## 📝 License

This project is for educational purposes. Consult legal and medical professionals before any clinical use.