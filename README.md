## Arctic Snow Depth on Sea Ice
### Prediction Model by Machine Learning Methods
code.ipynb file uses Python based machine learning software to invert and predict Arctic snow depth. It uses information from AMSR2 microwave remote sensing data.

Our exploration mainly focuses on extremely small and time dependent training samples.

13 models were included, although only **KNN**, **Extra Trees**, and **TabNet** were chosen as the final adopted models. The code for other models is also provided for researchers to modify or reference. We hope that researchers can explore more potential in small sample machine learning inspired by this code.

We provide some sample training data for model training demonstration. Specific information about these sample data can be found on the following website.

**pass**

Based on these data obtained models, we processed Arctic sea ice data **from 2013 to 2023**.



Researchers can modify the data preprocessing code to adapt to their own research data, mainly using the label processing function 'raw2process'.

The Python libraries required to ensure the proper functioning of the code are displayed in the **requirements.txt** file.
