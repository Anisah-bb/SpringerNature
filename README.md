# Junior MLOps Engineer Introduction Task

In this file, you will find a description of the task and the instructions to complete it.
Please read the instructions carefully and complete the task to the best of your ability.

A few important notes before you start:

1. You are free to use Google or any other resources you need to complete the tasks. We understand that research is a
   part of the problem-solving process.
2. You should be able to explain your code and the decisions you made. We will have a chat about your code after you
   have completed the task.
3. Everything should be able to run locally on your pc/laptop. We will not use any cloud services for this task.
4. The task is focused on learning to apply a library or framework to existing code. It is therefore important that you
   have a good understanding of the code you write, and also the code that is provided.
5. The task should take a maximum of 1 hour to complete. If you are not able to complete the task in this time, don't
   worry! Please submit what you have done, and we can discuss the rest in the next meeting.

## Task: Model Serving API

During research or development, it is common to use Machine Learning models in Python scripts.
But, to use a model in a real world environment, it is often necessary to serve the model through an API.
For this task, you will have to create a simple API to serve the model code that is provided.

You are given a pre-trained model which can embed texts and compute the similarities between two texts.
The task is to create a serving API, such that the model can be used without the need of Python code.

The code for the model is provided in the file `model.py`.
This file contains a class `Handler`, which can be used to load the model and use it for inference.
The handler already contains the code to embed an input text, or to calculate the similarity between two texts.
Investigate the code in the file to understand how to use the handler.

You will have to write an API to serve the embedding and similarity functionalities of the model.
This makes it possible for external users to use the model without the need of Python code.
To write the API, you have to use the FastAPI framework, which is a very easy to use framework for building APIs.
See the [FastAPI documentation](https://fastapi.tiangolo.com/) for more information on how to use it.
Their documentation contains a lot of examples and information on how to use the framework.

### Technical requirements

The model handler has two functions, `embed` and `similarity`, which should be served through the API.
The API should have two endpoints, one for each of these functions.

Using FastAPI, create an app with the following endpoints:

- `/embed`: a **POST** request to this endpoint should accept a JSON body containing a single text. The response should
  be a JSON body containing the embedding of the text.
- `/similarity`: a **POST** request to this endpoint should accept a JSON body containing two texts. The response should
  be a JSON body containing the similarity score (a float value) between the two texts.

Inspect the code in `model.py` and align the API input and output to the input and output of the model handler.

The code for the API app should be put in the file `app.py`, and should be able to run using the command
`python app.py` in a terminal.
There is already some code in place to start the app, but you will have to add the app itself and the endpoints.
If the app is running correctly, you can check http://localhost:8000/docs to see the documentation of the API and test
the endpoints.
(See [the documentation](https://fastapi.tiangolo.com/tutorial/first-steps/#interactive-api-docs) for more information)

## Deliverables

There are two ways to submit the task

### GitHub/GitLab

If you have a GitHub/GitLab account, you can create a repository and push the code to it.
Please add the following users as collaborators to the repository:

- https://github.com/Formsma
- https://github.com/pringled

and email `job.formsma@springernature.com` with the link to the repository.

### Email

If you do not have a GitHub/GitLab account, you can also send the code as a zip file.
To do this, create a zip file containing the code and any other files you want to submit.

Please email the zip file to

- `job.formsma@springernature.com`
- `thomas.vandongen@springernature.com`

## Questions

For questions about the task (can be any simple question!), please reach out to:

- `job.formsma@springernature.com`