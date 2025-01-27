## Arctic Snow Depth on Sea Ice
### Prediction Model by Machine Learning Methods
code_ML.ipynb file uses Python based machine learning software to invert and predict Arctic snow depth. It uses information from AMSR2 microwave remote sensing data.


Our exploration mainly focuses on extremely small and time dependent training samples.


13 models were included, although only **KNN**, **Extra Trees**, and **TabNet** were chosen as the final adopted models. The code for other models is also provided for researchers to modify or reference. We hope that researchers can explore more potential in small sample machine learning inspired by this code.


This code is only for demonstration purposes. Researchers can modify the data preprocessing code to adapt to their own research data, mainly changing the label processing function 'raw2process'.


The Python libraries required to ensure the proper functioning of the code are displayed in the **requirements.txt** file.

