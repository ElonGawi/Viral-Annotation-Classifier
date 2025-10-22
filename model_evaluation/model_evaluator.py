from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from typing import Protocol, runtime_checkable


class ModelReport(object):
    def __init__(self):

        self.metrics = None
        
        # confusion matrix and its displat labels 
        self.cm = None
        self.cm_display_labels = None

    def show_report(self):
        print(self.metrics)
        
        # create confustion matrix and display it
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, 
                                         display_labels=self.cm_display_labels)
        cm_disp.plot()


class ModelEvaluator(object):
    def __init__(self, model, eval_dataset):
        """Create model evaluators 

        Args:
            model (ModelEvalWrapper): model to evaluate
            eval_dataset (Dataframe): dataset to evaluate on, should have X and y 
        """
        assert isinstance(model, ModelEvalWrapper), "Model must be of a ModelEvalWrapper class"
        self.model = model
        self.eval_dataset = eval_dataset

    def _predict(self): 
        return self.model.model.predict(self.eval_dataset["X"])

    def generate_report(self):
        y_true = self.eval_dataset["y"]
        y_pred = self._predict()
        
        report = ModelReport()     
#        report.metrics = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        report.metrics = classification_report(y_true=y_true, y_pred=y_pred)

        # confustion matrix
        report.cm_display_labels = unique_labels(y_true, y_pred)
        report.cm = confusion_matrix(y_true,y_pred)
        return report
    

@runtime_checkable
class ModelEvalWrapperInterface(Protocol):
    """
    An interface for the model evaluator. 
    Implementing this interface is required for evaluating the model using ModelEvaluator
    """
    def predict(X):
        """
        :param X: a dataframe with X to prredict
        :returns the predictions as a dataframe
        """
        pass


class ModelEvalWrapper(object):
    """
    Wrapping the pretrained model to send for evaluation
    """
    def __init__(self, model, title):
        """
        :param model: Model to evaluate, the model should be already trained and implement the methods in  
        :param title: Title or label for this evaluation instance
        """
        assert isinstance(model, ModelEvalWrapperInterface), "model must implement the ModelEvalWrapperInterface Protocol"
        self.model = model
        self.title = title

