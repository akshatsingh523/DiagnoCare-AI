---
# DiagnoCare AI

**DiagnoCare AI** is an innovative healthcare solution that leverages advanced machine learning algorithms and natural language processing (NLP) to provide accurate disease predictions and personalized precautionary measures. This project aims to empower users with early diagnosis and actionable health recommendations through AI-driven insights.

## Key Features

- **Machine Learning Models**: We initially experimented with various models including Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Logistic Regression. After rigorous testing, we concluded that **Random Forest** provided the best results for accurate disease prediction.
  
- **Hyperparameter Tuning**: To further improve model performance, we implemented **Grid Search CV**, allowing us to fine-tune the hyperparameters for optimal accuracy. The final model is saved in both **h5** and **pickle** formats for flexibility, enabling easy integration or further fine-tuning.

- **Language Model Fine-Tuning**: For enhancing NLP capabilities, we utilized the **Llama 3.1 8B** model. The fine-tuning process was accelerated using **Unsloth**, allowing rapid customization of the model for precise and efficient healthcare recommendations.

- **Custom Dataset**: To ensure the model performs in a way that aligns with our specific requirements, we developed and trained it on a **custom dataset**. This allows for accurate disease prediction and context-aware precautionary measures, tailored to individual user inputs.

## How It Works

1. **Symptom Analysis**: Users input their symptoms, and the system uses the optimized Random Forest model to predict potential diseases.
2. **Disease Prediction**: Based on the symptoms, DiagnoCare AI delivers accurate predictions, helping users identify possible health conditions early.
3. **Precautionary Measures**: For each disease prediction, the platform offers tailored precautionary steps, empowering users to take preventive action.

## Tools & Technologies

- **Random Forest** (Optimal model for disease prediction)
- **Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Logistic Regression** (Initial models tested)
- **Grid Search CV** (For hyperparameter optimization)
- **Llama 3.1 8B** (Fine-tuned for natural language understanding and recommendation generation)
- **Unsloth** (Accelerating the fine-tuning process)

## Future Improvements

- Expanding the dataset for broader disease coverage
- Enhancing real-time interaction capabilities
- Integrating speech recognition for symptom input

## Conclusion

DiagnoCare AI is a powerful, AI-driven tool designed to improve early disease detection and provide personalized health recommendations. By leveraging machine learning and NLP, the platform offers a proactive approach to healthcare, enabling better health outcomes for users.
---
