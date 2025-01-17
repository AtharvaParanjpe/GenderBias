# Suicidal Risk

The main idea behind the research is to analyse a bias if present by race, gender, or location in people with suicidal tendancies.
For this analyses we chose the publically available Synthetic Mass dataset.


## 1. Synthea Dataset 

Data hosted within SyntheticMass has been generated by SyntheaTM, an open-source patient population simulation made available by The MITRE Corporation.

The data is free from cost, privacy, and security restrictions. It can be used without restriction for a variety of secondary uses in academia, research, industry, and government.

## 2. Data Preparation

The type of dataset that was used was of CSV format. 8 files contained records of patients and their visits for different reasons. These records were read file by file and filtered by the keywords : "self harm" and "sui" which were said to indicate the visit being for suicidal patients. These were then matched with the data of the patients in patients.csv using the patient ids and assigned a target value of '1'.

Since the data was large, the entire data was split into 12 different files. As a result, a unique filter was applied. Finally, the data was appended at the end of the patients file.

A similar approach was used to take non suicidal patients randomly and these were appended with a target '0' to account for the negative samples.

Another major step involved recording the previous visits of patients who were deemed suicidal so as to analyse what factors colud contribute to a persons suicidal tendancy.

## 3. Preparing a model

The patients dataset was cleaned by removing unnecessary columns like the patient id and the remaining ones were label encoded. 
Now to judge bias, we first chose to evaluate our model on gender. Therefore, the entire data was split into male and female patients. To balance the number of records in each we used the smote library.

We then evaluated each group of dataset on three different models namely, Linear Regression, Logistic Regression and Support Vector Machines. For each group (male or female), the following scores were computed: 
- True Positive rate , tpr
- False Positive rate, fpr
- True Negative rate, tnr
- False Negative rate, fnr
- Accuracy
- Area under curve, auc

## 4. Results

The "Final_File_before_label_encoding.xlsx" represents the resulting data with feature removal.

The "Final_File_min13.xlsx" represents the final data being sent to the model.

The results can be found in "Result_with_min_13.xlsx".  

