import time
import psutil
import os
import tracemalloc, time
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from typing import Protocol, runtime_checkable
import threading
import matplotlib.pyplot as plt


class ModelReport(object):
    def __init__(self):
        
        self.model_title = None
        self.metrics = None
        
        # confusion matrix and its displat labels 
        self.cm = None
        self.cm_display_labels = None
        
        ## model performence 
        # how long it took to predict 
        self.model_runtime = None
        self.avg_time_per_prediction = None

        # memory profling 
        self.memory_records = None

    def show_report(self):
        print(f"#####\t Report for Model: f{self.model_title}\t\n")

        print(self.metrics)
    
        print(f"The model took {self.model_runtime:.5f} seconds to run\n")
        print(f"Average time per prediction {self.avg_time_per_prediction:.5f} seconds\n")
        
        
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
    

    def _monitor_memory(interval=0.01):
        """Continuously record memory usage of the current process."""

        # select current process
        process = psutil.Process()
        
        memory_records = {}
        start_time = time.time()
        # add a record for the memory at time t=0
        memory_records[0] = process.memory_info().rss

        def record():
            while not stop_event.is_set():
                mem = process.memory_info().rss # in bytes
                current_timestamp = time.time() - start_time # timestamp is the time delta
                memory_records[current_timestamp] = mem
                time.sleep(interval)

        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=record)
        monitor_thread.start()
        return stop_event, memory_records


    def _predict_and_profile(self, report, monitor_mem=True, mem_monitor_interval=0.01):
        """run preidtc and track performence such as time, mem usage

        Args:
            report (_type_): report to fill in
        """
        X_len = len(self.eval_dataset["X"])
        
        if monitor_mem:
            stop_event, memory_records = ModelEvaluator.monitor_memory(mem_monitor_interval)
        
        start_time = time.perf_counter()

        preds = self._predict()

        end_time = time.perf_counter()

        if monitor_mem:
            stop_event.set()
            report.memory_records = memory_records

        report.model_runtime = end_time - start_time # seconds 
        report.avg_time_per_prediction = report.model_runtime/X_len

        return preds, report

    def generate_report(self):
        report = ModelReport()     
        report.model_title = self.model.title

        y_true = self.eval_dataset["y"]

        y_pred, report = self._predict_and_profile(report=report)

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

