# Data-Driven Consumer Analytics

## Introduction

In this project, converted a sample real-time data and developed an AI-driven analytical solution, leveraging machine learning to extract valuable business insights, segment customers efficiently, and identify lookalike consumers for targeted marketing and engagement.

## Objectives

The primary goal of this project is to perform :
   1. **Exploratory Data Analysis (EDA) & Business Insights** – Uncover trends in customer behavior, product performance, and regional sales to drive data-backed decision-making.
   2. **Customer Segmentation** – Apply clustering techniques (K-Means) to analyze customer spending behaviour and find out **High Value Customers** from data.
   3. **Lookalike Consumer Modeling** – Use TF-IDF and Cosine Similarity to identify customers with similar shopping behaviors, enabling better targeting and engagement.

### Prerequisites
To run this project, you need to install the following libraries:
### Required Libraries

- **Python 3.12+**
- **Pandas**: This library performs data manipulation and analysis also provides powerful data structures like dataframes.
- **Scikit-Learn**: Scikit-learn library provides tools for machine learning, including classification, regression, clustering, and dimensionality reduction.
- **Streamlit**: Streamlit is a framework that builds interactive, data-driven web applications directly in python. 

Other Utility Libraries : **Matplotlib**, **io**.

### Installation

   ```
   pip install pandas
   pip install streamlit
   pip install scikit-learn
   pip install matplotlib
   ```

### Procedure

1.   Create new directory **'Consumer Analytics'**.
2.   Inside that directory/folder create new environment.
   
   ```
   python -m venv aica
   ```

  Now, activate this **'aica'** venv.
  
4.   Clone this Repository :

   ```
   https://github.com/Rajcr2/Data-Driven-Consumer-Analytics.git
   ```
5.   Now, Install all mentioned required libraries in your environment.
6.   After, that Run **'dashboard.py'** file from Terminal. To activate the dashboard on your browser.
   ```
   streamlit run dashboard.py
   ``` 
7. Now, move to your browser and Analyse this Insights from Web Dashboard.


## Output

### EDA & Business Insights 

![image](https://github.com/user-attachments/assets/32c9233d-1aab-4c3a-bf10-84368f0bbdbf)

![image](https://github.com/user-attachments/assets/a2a15468-66c9-4d8d-93a5-137fcc14c002)

### Customer Segmentation

![image](https://github.com/user-attachments/assets/30ed6768-be6e-4025-b9c9-2ab0c89b0a57)


### High-Value Customers List :
![image](https://github.com/user-attachments/assets/ab9f7b65-3a49-4515-8dc6-b35f4a0936df)



### Lookalike Consumer Modeling 

![image](https://github.com/user-attachments/assets/ce8119b5-26d6-4aae-8716-3ec7131eb066)


### Conclusion 

Like this we can develope a Consumer Analytics application which can provide same Business Insights and other things where we just have pass the data into that application.
Thats all, Thanks for Reading stay tuned for More.



