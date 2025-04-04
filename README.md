# smart-water-management-system
Project Title: Water Consumption Prediction Using XGBoost

Project Overview:
This project focuses on predicting water consumption based on various features such as apartment type, income level, and environmental factors. Using machine learning, specifically XGBoost, we aim to develop a robust predictive model that helps in optimizing water usage and planning resources efficiently.

Technologies Used:
- Python: For data preprocessing, feature engineering, and model training.
- Pandas: For handling and processing structured data.
- NumPy: For numerical computations.
- Scikit-learn: For data splitting and evaluation metrics.
- XGBoost: For building an optimized regression model.
- GPU Acceleration: Using CUDA to enhance model training speed.

Key Steps in the Project:
1. Data Preprocessing:
   - Loaded training and test datasets.
   - Cleaned and formatted feature names.
   - Handled missing values using median imputation.
   - Converted categorical features using One-Hot Encoding.
   - Converted humidity values from percentage to numeric format.
   
2. Feature Engineering:
   - Ensured consistency in feature columns between train and test data.
   - Dropped unnecessary columns such as Timestamp from training features.

3. Model Training:
   - Split data into training and validation sets (80-20 split).
   - Used XGBoost’s DMatrix format for efficient memory usage.
   - Configured hyperparameters (learning rate, max depth, regularization) to optimize performance.
   - Utilized GPU acceleration (tree_method=hist, device=cuda) for faster computation.
   - Implemented early stopping to prevent overfitting.

4. Model Evaluation:
   - Used Mean Squared Error (MSE) as the evaluation metric.
   - Applied custom scoring: max(0, 100 - sqrt(MSE)).
   - Achieved a validation score close to 90 by fine-tuning hyperparameters.

5. Prediction and Submission:
   - Used the trained model to predict water consumption for the test dataset.
   - Generated a CSV file with Timestamp and Water_Consumption as the final submission.

Project Outcome:
This project successfully implemented a machine learning pipeline for water consumption prediction, achieving high accuracy and efficiency using GPU-powered XGBoost. The model can be further improved with additional data and feature engineering techniques.
