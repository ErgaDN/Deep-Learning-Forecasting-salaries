# Predictive Model for Data Science Salaries
### Abstract
This project aims to develop a robust model for predicting the salaries of data science professionals. Initial attempts using linear regression models yielded unsatisfactory results, leading to the exploration of more complex models including neural networks and LSTM (Long Short-Term Memory) models. The final approach, combining LSTM with additional tuning and feature engineering, achieved a promising RMSE (Root Mean Square Error) of 2.477, indicating a significant improvement in predictive accuracy.

### Introduction
With the rapid growth of the Data Science field, understanding the dynamics of salary variations is crucial for professionals and employers alike. This project explores various predictive models to estimate salaries based on job-related features, moving from basic linear regression to advanced deep learning techniques.

### Methodology
## Data Preparation
The dataset was initially preprocessed for numerical stability and feature encoding.
Features such as the ratio of 'salary in USD' to 'salary', experience level, and normalized percentile ranks within job categories were calculated.
The data was split into training and testing sets, with numerical features standardized.
## Model Architecture
Multiple models were tested, including linear regression, neural networks, LSTM, and CNN models.
The LSTM model included an input layer, two LSTM layers with dropout for regularization, and a dense output layer.
## Training
The LSTM model was trained with an Adam optimizer over 100 epochs, using mean squared error as the loss function.
## Results
The LSTM and CNN models showed the most promise, with RMSEs of 2.48 and 2.73, respectively.
Feature importance analysis revealed key roles significantly influencing salary predictions.
## Conclusion and Future Work
The study demonstrates the effectiveness of LSTM and CNN models in predicting data science salaries. Future work could explore further optimization, additional features, and expanded datasets to enhance model performance.
