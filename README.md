## Arctic Spring Snow Depth
### Background
The theoretical framework for snow depth retrieval follows Markus and Cavalieri [1] and Rostosky, et al. [2], and the input parameters include: $TB_{ice} (7V)$,  〖TB〗_ice (19V), 〖TB〗_ice (37V), 〖GR〗_ice (19V/7V),〖GR〗_ice (37V/19V), and 〖PR〗_ice (37)


The training reference was obtained from the latest version of the AWI IceBird airborne snow depth data, available from 10.1594/PANGAEA.966009 (Apr 2017) [3] and  10.1594/PANGAEA.966057(Apr 2019) [4]


### Introduction to Code
code_ML.ipynb file uses Python-based machine learning methods to invert and predict Arctic spring snow depth.


Our exploration mainly focuses on relatively small and time independent training samples.


13 models were included, although only **KNN**, **Extra Trees**, and **TabNet** were chosen as the final adopted models. The code for other models is also provided for researchers to modify or refer to. We hope that researchers can explore more potential in small sample machine learning inspired by this code.


This code is only for demonstration purposes. Researchers can modify the data preprocessing code to adapt to their own research data, mainly changing the label processing function 'raw2process'.


The Python libraries required to ensure the proper functioning of the code are displayed in the **requirements.txt** file.


### Results


Based on **KNN**, **Extra Trees**, and **TabNet** models, snow depth prediction results are shown below:


**Spring snow depth over pan-Arctic sea ice derived from AMSR2 using machine learning methods (2013-2023)**
Yi Zhou, Chentong, Zhang, & Xianwei Wang. (2025). Spring snow depth over pan-Arctic sea ice derived from AMSR2 using machine learning methods (2013-2023) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14739244


### References

[1] T. Markus and D. J. Cavalieri, “Snow depth distribution over sea ice in the Southern Ocean from satellite passive microwave data,” Antarctic sea ice: physical processes, interactions and variability, vol. 74, pp. 19-39, Jan. 1998.
[2] P. Rostosky, G. Spreen, S. L. Farrell, T. Frost, G. Heygster, and C. Melsheimer, “Snow Depth Retrieval on Arctic Sea Ice From Passive Microwave RadiometersImprovements and Extensions to Multiyear Ice Using Lower Frequencies,” Journal of Geophysical Research: Oceans, vol. 123, no. 10, pp. 7120-7138, Sep. 2018.
[3] A. Jutila, S. Hendricks, R. Ricker, L. von Albedyll, and C. Haas, “Airborne sea ice parameters during the PAMARCMIP2017 campaign in the Arctic Ocean, Version 2,” PANGAEA, 2024, doi: 10.1594/PANGAEA.966009.
[4] A. Jutila, S. Hendricks, R. Ricker, L. von Albedyll, and C. Haas., “Airborne sea ice parameters during the IceBird Winter 2019 campaign in the Arctic Ocean, Version 2,”PANGAEA, 2024, doi: 10.1594/PANGAEA.966057.
