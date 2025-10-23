import pandas as pd
import time
import psutil
import os
import tracemalloc, time
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from typing import Protocol, runtime_checkable
from models.config import AnnotationLabels
import threading
import matplotlib.pyplot as plt


class ModelReport(object):
    def __init__(self):
        
        self.model_title = None
        self.metrics = None
        
        # confusion matrix and its display labels 
        self.cm = None
        self.cm_display_labels = None
        
        ## model performence 
        # how long it took to predict 
        self.model_runtime = None
        self.avg_time_per_prediction = None

        # memory profling 
        self.memory_records = None

    def show_report(self):
        print(f"#####\t Report for Model: {self.model_title}\t\n")

        print(self.metrics)
    
        print(f"The model took {self.model_runtime:.5f} seconds to run\n")
        print(f"Average time per prediction {self.avg_time_per_prediction:.5f} seconds\n")
        
        
        # create the subplot layout
        fig, axes = plt.subplots(3, 1, figsize=(6, 10))  # figsize optional
        # the axis the plot will use
        cm_plot = axes[0]
        mem_plot = axes[1]
        cpu_plot = axes[2]

        ### create Confusion matrix and display it
        cm_plot.set_title("Confusion Matrix")
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, 
                                            display_labels=[AnnotationLabels.id2label[i] for i in self.cm_display_labels]
                                            )
        cm_disp.plot(ax=cm_plot)


        ### the mem usage plot
        mem_records_df = pd.DataFrame(list(self.memory_records.items()), columns=["TimeDelta", "MemUsage"])
        # only ge mem usage delta
        mem_usage_t0 = mem_records_df[mem_records_df["TimeDelta"] == 0]["MemUsage"][0]
        mem_records_df["MemUsageDelta"] = mem_records_df["MemUsage"] - mem_usage_t0
        # convert mem usage to MB
        mem_records_df["MemUsageDeltaMB"] = mem_records_df["MemUsageDelta"] / (1024 * 1024)

        mem_plot.plot(mem_records_df["TimeDelta"], mem_records_df["MemUsageDeltaMB"])
        mem_plot.set_title("Memory Usage")
        mem_plot.set_xlabel("Time (seconds)")
        mem_plot.set_ylabel("Delta memory usage(MB)")

        
        ### Cpu plot        
        cpu_records_df = pd.DataFrame(list(self.cpu_usage_records.items()), columns=["TimeDelta", "CPUUsage"])

        cpu_plot.plot(cpu_records_df["TimeDelta"], cpu_records_df["CPUUsage"])
        cpu_plot.set_title("CPU Usage across all cores")
        cpu_plot.set_xlabel("Time (seconds)")
        cpu_plot.set_ylabel("CPU Usage (%) of cores (100% = 1 core)")


        plt.tight_layout()
        plt.show()



class ModelEvaluator(object):
    def __init__(self, model, eval_dataset):
        """Create model evaluators 

        Args:
            model (ModelEvalWrapper): model to evaluate
            eval_dataset (Dataframe): dataset to evaluate on, should have 2 columns "protein_annotation"  and "label"
        """
        assert isinstance(model, ModelEvalWrapper), "Model must be of a ModelEvalWrapper class"
        self.model = model
        self.eval_dataset = eval_dataset


    def _predict(self): 
        return self.model.model.predict(self.eval_dataset["protein_annotation"])
    

    def _profile(interval=0.01):
        """Continuously record memory usage of the current process."""

        # select current process
        process = psutil.Process()
        
        memory_records = {}

        # keys are delta t, and the value is the CPU usage in precentage 
        # (1 core = 100%, can be over 100% if multiple cores are used) 
        # since the last call to the function so practically this is delta CPU usage
        cpu_usage_records = {} 
        
        start_time = time.time()

        # add a record for the memory at time t=0
        memory_records[0] = process.memory_info().rss
        
        # since this is the delta usage, first call will return 0, see https://psutil.readthedocs.io/en/latest/index.html#psutil.Process.memory_maps
        cpu_usage_records[0] = process.cpu_percent(interval=None)

        def record():
            while not stop_event.is_set():
                mem = process.memory_info().rss # in bytes
                current_timestamp = time.time() - start_time # timestamp is the time delta
                memory_records[current_timestamp] = mem
                cpu_usage_records[current_timestamp] = process.cpu_percent(interval=None)
                time.sleep(interval)

        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=record)
        monitor_thread.start()
        return stop_event, memory_records, cpu_usage_records


    def _predict_and_profile(self, report, profile=True, monitor_interval=0.01):
        """run preidtc and track performence such as time, mem usage

        Args:
            report (_type_): report to fill in
        """
        X_len = len(self.eval_dataset["protein_annotation"])
        
        if profile:
            stop_event, memory_records, cpu_usage_records = ModelEvaluator._profile(monitor_interval)
        
        start_time = time.perf_counter()

        preds = self._predict()

        end_time = time.perf_counter()

        if profile:
            stop_event.set()
            report.memory_records = memory_records
            report.cpu_usage_records = cpu_usage_records

        report.model_runtime = end_time - start_time # seconds 
        report.avg_time_per_prediction = report.model_runtime/X_len

        return preds, report

    def generate_report(self):
        report = ModelReport()     
        report.model_title = self.model.title

        y_true = self.eval_dataset["label"]

        y_pred, report = self._predict_and_profile(report=report)

#        report.metrics = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        report.metrics = classification_report(y_true=y_true, y_pred=y_pred)

        # Confusion matrix
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

