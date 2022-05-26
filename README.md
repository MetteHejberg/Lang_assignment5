## 1. Assignment 5 - self-assigned: age classifier 
Link to repository: https://github.com/MetteHejberg/Lang_assignment5

This assignment is inspired by Sagae's (2021) paper, where they built a recurrent neural network that tracks children's age and syntactic development. The goal of their paper, was to provide a more data-driven and reproducible alternative to the models currently in use to predict and evaluate children's age from their language skills. This assignment seeks to build a classifier that can predict children's age on the basis of their language skills.

I got the data from the CHILDES corpus (MacWhinney, 2022). CHILDES is a child language corpus of naturally produced speech. Most data on the site are audio recordings which also have been transcribed. I downloaded transcriptions of children between the ages 1 and 5 from the following providers on the site:
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
A large part of this assignment was to wrangle the data to get them into a format that I could work with. Firstly, the script loads the data as ```pandas``` dataframes. This allows me to work with the files very efficiently eventhough they are not csv files. Then, for every dataset, I get what the child says and the metadata on the child (age and sex), which I collectively tranform into a new dataframe. I then use regex to clean the text and age columns. I rename the columns and drop data from 1 and 5-year-olds because there is less data, and . Lastly, I create a train and test split with ```scikit-learn```. I use the tokenizer from ```tensorflow``` to tokenize the text and the ```sequence``` function to further process the text. I both use the label binarizer and encoder functions. I build a sequential model with user-defined paramters from the command line. Finally, I plot the output of the loss function and the accuracy, which I save, and create the classification report, which I also save.

## 3. Usage ```nlp_age_classifier.py```
To run the code you should:
- Pull this repository with this folder structure
- Retrieve the data and place it in a folder called ```data``` inside the folder ```in```
- Make sure the ```utils``` folder is placed inside ```src```
- Install the libraries mentioned in ```requirements.txt``` from the command line
- Set your current working directory to the level above ```src```
- Write in the terminal: ```python src/age_classifier.py -e "number of epochs of the model" -b "batch size of the model```
  - The outputs in ```out``` were created with the following code: ```python src/age_classifier.py -e 10 -128```

## 4. Discussion of Results 
The outputs of the models are vastly different. The plot of the unbalanced model shows that the model overfits on the training data beyond 5 epochs, and however achieves around 60% accuracy overall, with the best performance on the 2 year-olds by far. Interestingly, the model performs extremely badly on the 4-year-olds. This is perhaps not surprising, since there are vastly fewer tokens in this class than the others. In general, there are huge difference in the amount of tokens. I already tried to combat this before I created the model by excluding the data from 1 and 5-year-olds on the grounds at there wasn't enough data. On the basis of the outpus of the unbalanced model, I decided to create a balanced dataset and run the same model again. The outputs of this run of the model show that the training and test data follow each other very closely, and so balancing the data removed the first issue with the previous run of the model. The model still performs best to predict 2-year-old speech. However, after balancing the data, the accuracy of this class has gone down, and the model succeeded at predicting 4-year-old speech much better. Interestingly, the 3-year-old speech became more difficult to predict. The model performs at an overall accuracy of and 50%, which means that although balancing the data helped the model perform better on one class, it is a trade-off of losing accuracy in general, because this model clearly needs a lot of data in order to perform well. 

This assignment required a lot of wrangling using regex. One way I could perhaps improve the code and therefore improve the performance of the model, is to use a negative match for the regex instead. So instead of defining what I wish to remove, I could define what I wanted to keep of the text. There are so many rows in the dataset that it is impossible to catch all unwanted strings with the regex solution I have used, and this could potentially have impacted the results.

Lastly, I included epochs and batch size as user-defined arguments from the command line, however this means, from how the code is written, that both runs of the model will use the same parameters. While this allows for a more direct comparison of the model’s performance, since I only balanced the dataset, exactly because I balanced the dataset, the parameters set from the command line might not be the best match for the new balanced dataset. This of course goes both ways, so the user could define parameters that are optimal for the balanced but not the unbalanced dataset. I therefore encourage the user to try a range of different parameters. Recall that the model did not overfit on the training data on the balanced dataset. It would therefore be interesting to add more epochs to see if that could improve the accuracy, or the model will start overfitting. Importantly, I did not add the learning rate as a user-defined parameter, because I found that the one I set in the script to be the best for both runs of the model.

