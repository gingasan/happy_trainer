import numpy as np
from sklearn.metrics import accuracy_score
 
 
def compute_metrics_exact_match(eval_outputs):
    predictions, labels = eval_outputs
    labels = labels[:, 1:]
    predictions = predictions[:, :-1]
    score = 0
    for i in range(labels.shape[0]):
        l, p = labels[i], predictions[i]
        if np.array_equal(p[l != -100], l[l != -100]):
            score += 1
    score /= predictions.shape[0]
    return {"exact_match": score}
 
def compute_metrics_accuracy(eval_outputs):
    predictions, labels = eval_outputs
    labels = labels[:, 1:].reshape(-1)
    predictions = predictions[:, :-1].reshape(-1)
    score = accuracy_score(y_true=labels[labels != -100], y_pred=predictions[labels != -100])
    return {"accuracy": score}
 
 
def grid_search_plan_a(args):
    for config in [
        {
            "learning_rate": 1e-5,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 3
        },
        {
            "learning_rate": 1e-5,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 3
        }
    ]:
        args.learning_rate = config["learning_rate"]
        args.train_batch_size = config["train_batch_size"]
        args.gradient_accumulation_steps = config["gradient_accumulation_steps"]
        args.num_train_epochs = config["num_train_epochs"]
        yield args
