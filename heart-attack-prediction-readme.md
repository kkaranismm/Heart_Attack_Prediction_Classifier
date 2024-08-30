# Heart Attack Prediction using XGBoost

## Project Overview

This project aims to predict the risk of heart attacks using machine learning, specifically the XGBoost (Extreme Gradient Boosting) algorithm. By leveraging a comprehensive dataset of patient information, including medical history, lifestyle factors, and demographic details, we've developed a powerful tool to assist healthcare professionals in identifying individuals at higher risk of heart attacks.

## Motivation

Heart disease remains a leading cause of mortality worldwide. Early detection and prevention of heart attacks are crucial for improving patient outcomes and saving lives. This project addresses this critical need by harnessing the power of machine learning to create a smart system for heart attack risk prediction.

## Technologies Used

### Frontend:
- HTML: Provides the structure for the web interface.
- CSS: Styles the web page for a user-friendly experience.
- JavaScript: Adds interactivity, including a parallax effect on the background image.

### Backend:
- Flask: Powers the web application, handling server-side operations.
- Pickle: Used for saving and loading the trained machine learning model.
- Python: The primary programming language for backend development and data processing.

### Machine Learning:
- pandas: For data manipulation and analysis.
- scikit-learn: Utilized for data splitting, model evaluation, and preprocessing.
- XGBoost: The core machine learning algorithm used for prediction.
- numpy: For numerical computing and array operations.

## Dataset

The dataset used in this project is derived from the Cleveland database and includes 14 key attributes related to heart health:

1. Age
2. Sex
3. Chest Pain Type
4. Resting Blood Pressure
5. Serum Cholesterol
6. Fasting Blood Sugar
7. Resting Electrocardiographic Results
8. Maximum Heart Rate Achieved
9. Exercise Induced Angina
10. Oldpeak (ST Depression)
11. Slope of Peak Exercise ST Segment
12. Number of Major Vessels Colored by Fluoroscopy
13. Thal (Thallium Stress Test Result)
14. Target (0 = low risk, 1 = high risk of heart attack)

## Methodology

1. **Data Collection and Exploration**: Gathered and analyzed the dataset to understand its structure and relationships between variables.
2. **Data Preprocessing**: Handled missing data, encoded categorical variables, and scaled numerical features.
3. **Model Selection**: Chose XGBoost for its performance in similar scenarios and resistance to overfitting.
4. **Training**: Split the data into training and testing sets, then trained the XGBoost model on the training data.
5. **Evaluation**: Assessed the model's performance using metrics such as accuracy, precision, recall, and F1-score.
6. **Hyperparameter Tuning**: Optimized the model's hyperparameters to improve performance.
7. **Deployment**: Integrated the model into a Flask web application for easy use.

## Results

- The model achieved an impressive accuracy of 98.53% in predicting heart attack risk.
- Visualizations include distribution of heart attack risk by age, confusion matrix, and ROC curve.
- The web application provides an intuitive interface for users to input their medical data and receive instant risk predictions.

## How to Use

1. Clone the repository
2. Install the required dependencies (listed in `requirements.txt`)
3. Run the Flask application
4. Access the web interface through your browser
5. Input the required medical information
6. Receive an instant prediction of heart attack risk

## Future Work

- Incorporate more diverse datasets to improve model generalization
- Explore other machine learning algorithms for comparison
- Enhance the web interface with more detailed risk explanations and preventive recommendations
- Collaborate with healthcare professionals for real-world validation and implementation

## Contributors

[List of contributors]

## License

[Specify the license]

## Acknowledgments

- Cleveland Clinic Foundation for the heart disease dataset
- Anthropic for AI assistance in project development and documentation

For more detailed information, please refer to the full project report included in this repository.
