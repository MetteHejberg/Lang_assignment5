## 1. Assignment 5 - self-assigned: age classifier 
Link to repository: https://github.com/MetteHejberg/Lang_assignment5

This assignment is inspired by Sagae's (2021) paper, where they build a recurrent neural network that tracks children's age and syntactic development. The goal of their paper, was to provide a more data-driven and reproducible alternative to the models currently in use to predict and evaluate children's age in terms of their language skills. This assignment seeks to build a classifier that can predict children's age on the basis of their language skills.

I got the data from the CHILDES corpus. CHILDES is a child language corpus of naturally produced speech. Most data on the site are audio recordings which also have been transcribed. I downloaded transcriptions of children between the ages 1 and 5 from the following providers on the site - __see how to cite them!__:
- Braunwald: Laura. https://childes.talkbank.org/access/Eng-NA/Braunwald.html
- Brown: Adam, Eve, and Sarah. https://childes.talkbank.org/access/Eng-NA/Brown.html
- Clark: Shem. https://childes.talkbank.org/access/Eng-NA/Clark.html
- Demetras1: Trevor. https://childes.talkbank.org/access/Eng-NA/Demetras1.html
- Kuczaj: Abe. https://childes.talkbank.org/access/Eng-NA/Kuczaj.html
- Sachs: Naomi. https://childes.talkbank.org/access/Eng-NA/Sachs.html
- Snow: Nathaniel. https://childes.talkbank.org/access/Eng-NA/Snow.html
- Suppes: Nina. https://childes.talkbank.org/access/Eng-NA/Suppes.html
- Weist: Benjamin and Jillian. https://childes.talkbank.org/access/Eng-NA/Weist.html

I tried to use the same data as Sagae (2021, p. 8), however there were a few that didn't fit my criteria. I wanted children within the ages of 1 and 5, primarily because these are the ages that there is most data from in the corpus and in the literature in general (Tomasello, 2003). This is therefore not to say that the linguistic development of children beyond the age of 5 is not interesting, but rather that we are missing data from age groups beyond 5 and these children's linguistic development is therefore vastly understudied. In fact, I only retrieved few files from children aged 5, but chose to include them because their linguistic skills are far more complex then children aged 3 and 4, and I therefore hoped the model would be able to predict this class inspite of the minimal data.

__References:__

MacWhinney, B. (2022). *The CHILDES Project: Tools for analyzing talk. Third Edition*. Mahwah, NJ: Lawrence Erlbaum Associates.

Sagae K. (2021). Tracking Child Language Development With Neural Network Language Models. *Frontiers in psychology*, 12, 674402. https://doi.org/10.3389/fpsyg.2021.674402

Tomasello, M. (2003). Constructing a language: *A usage-based theory of language acquisition*. Harvard University Press. 

## 2. Method
A large part of this assignment was to wrangle the data to get them into a format that I could work with. 

## 3. Usage ```nlp_age_classifier.py```
To run the code you should:
- Pull this repository with this folder structure
- Retrieve the data and place it in a folder called ```data``` inside the folder ```in```
- Install the libraries mentioned in ```requirements.txt``` from the command line
- Set your current working directory to the level above ```src```

## 4. Discussion of Results 
