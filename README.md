# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run
To train a specific model, simply run the command from the shell:
```shell
python main.py [options]
```
where options are:
* `-c`, `--config_file`: Path to the configuration file to be used. There are present one for LinearRegression (`default.json`) and one for `XGBoost`. Default is `./config/default.json`.

### Configuration file
The configuration files are the central part of the project. They outline all the choices and combinations that can be made to the pipeline.\
From there it's possible to include or exclude each part of the pipeline, making it possible to adjust the execution on the fly. This way it's possible to first disable the deploy and model training and just concentrate on the data preparation, controlling each step separately and having the possibility of finding many possible graphs saved inside the `train` folder, that contains a subfolder for each run, marked using the epoch as the id, making it possible to also have all the trainings ordered.


Although the configuration file tries to be as self-explanatory as possible, let's explain the main sections briefly.
- `data`: This first sections focuses on the data preparations part. It contains the `name` of the model, whether we want to retrieve the dataset from local or from an url and then various subsections for all the steps required to properly prepare the dataset.
    - `cleaning`: For data cleaning.
        - `dropColumns`: To drop specific columns.
        - `dropDuplicates`
        - `dropNa`
        - `dropCustom`: To define specific conditions to drop columns. If `rangeOrEqual` is `true` it searches for `min` and `max`, otherwise for `value`.
    - `exploration`: For data exploration, it's possible to define which figures we want to create. Last subsection `categorical` controls both violin plots and scatter plots by price.
    - `processing`: For processing the data based on the model we want to train. `default.json` and `xgb.json`, for example, use 2 different model and so need different processing steps.
- `model`: This section controls whether we want to train the model or not and has various subsections to controls the `evaluation` metrics, whether we want to save the model locally, if we want to produce a god figure, transform data and if we want to optimize the hyperparameters.
- `deploy`: Lastly the section that controls the deploy. From here it's possible to decide whether to deploy a Flask server or not and if we want to use a previously saved model or to train it right before deploying it. This is based on `trainOnTheSpot`.



## API
As prewviously said the API is available based on the configuration file settings. It is based on Flask and the project features a `request.http` file to be used in conjunction with Rest Client VS code extension.\
Following are the endpoints:

#### `POST /predict`
This predicts the value of a diamond based on its features. It takes as parameters:
* **carat**: [*float*]
* **cut**: [*string*]
* **color**: [*string*]
* **clarity**: [*string*]
* **depth**: [*float*]
* **table**: [*float*]
* **x**: [*float*]
* **y**: [*float*]
* **z**: [*float*]

#### POST `/similar`
This finds `n` entries in the dataset with the same cut, colour and clarity, and with the most similar weight.

Its parameters are:
* **n** [*integer*] The number of entries to be returned.
* **carat** [*float*] 
* **cut** [*string*]
* **color** [*string*]
* **clarity** [*string*]

#### GET `/interactions`
This returns the content of the database, to make it easier to consult it.

#### GET `/health`
Just to chech whether the server is up. Returns `Hello, Flask!`


## Train folder
The train folder contains all that is produced. Its structured based on subfolders, one for each iteration, marked by the epoch at the time of creation. This is better than using as UUID because it is possible to order them based on time.\
Each subfolder contains:
- The `config.json` file used at the time of creation **enriched with the metrics** produced during the training.
- The `model.pkl` file if the model was saved locally.
- Various `graphs` based on the ones that were chosen in the configuration.

## Conclusion
The project was really fun to work on.\
Would have loved to make many more improvements and I would be happy to discuss them and why I did not make them in the next interview if you please.