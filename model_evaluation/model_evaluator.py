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
from IPython.display import display, HTML
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
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display, HTML
import numpy as np
import io, base64
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display, HTML
import numpy as np
import io, base64
import json
import pickle
from dash import Dash, dash_table, Input, Output, dcc, html
import pandas as pd
import numpy as np


class ModelReport(object):
    def __init__(self):
        
        self.model_title = None
        self.model_info = None
        # self.metrics = None
        self.metric_per_label = None
        self.accuracy = None
        self.macro_avg = None
        self.weighted_avg = None

        # confusion matrix and its display labels 
        self.cm = None
        self.cm_display_labels = None
        
        ## model performence 
        # how long it took to predict 
        self.model_runtime = None
        self.avg_time_per_prediction = None

        # memory profling 
        self.memory_records = None

    def add_classfication_report_dict(self, metrics_dict):
        """ 

        Args:
            classification_report_dict (_type_): dict from sklearn classification_report
        """
        self.metric_per_label = {}
        all_label_ids = [AnnotationLabels.label2id[l] for l in AnnotationLabels.label_names]
        for label_id in all_label_ids:
            self.metric_per_label[AnnotationLabels.id2label[label_id]] = metrics_dict[str(label_id)]

        self.accuracy = metrics_dict["accuracy"]
        self.macro_avg = metrics_dict["macro avg"]
        self.weighted_avg = metrics_dict['weighted avg']

    def _display_titles(self):
        model_title = self.model_title

        HTML_title = f"""<div style="font-size:20px; line-height:1; margin-bottom:1px;">
                        <h2 style='margin-bottom:-10px;'>Performance Report: {model_title}</h2>
                        </div>"""

        # display(HTML(HTML_title))

        ###### Subtitle
        model_info = self.model_info
        HTML_subtitle = f"""<div style=" line-height:1; margin-bottom:1px;">
                        <h3 style='margin-bottom:1px;'>{model_info}</h2>
                        </div>"""

        # display(HTML(HTML_subtitle))
        return HTML_title, HTML_subtitle

    def _display_cm_abs_plot(self, return_img=True):
        
        # create the subplot layout
        fig, ax = plt.subplots()  # ax is the Axes

        # abs values
        ax.set_title("Confusion Matrix (Absolute values)")
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, 
                                            display_labels=[AnnotationLabels.id2label[i] for i in self.cm_display_labels]
                                            )

        cm_disp.plot(cmap="Blues", values_format=".1f", colorbar=False, ax=ax)
        
        if not return_img:
            plt.plot()
        else:
            cm_disp.plot(cmap="Blues", values_format=".1f", colorbar=False, ax=ax);

            buf = io.BytesIO()

            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)

            buf.seek(0)

            # Why is there plt.close() twice you ask, well me too. only way it works though otherwise pyjupter craete additional empty plot in cell
            plt.close()
            cm_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return cm_base64

    def _display_cpu_usage_plot(self):

        cpu_records_df = pd.DataFrame(list(self.cpu_usage_records.items()), columns=["TimeDelta", "CPUUsage"])

        time = cpu_records_df["TimeDelta"]
        cpu_usage = cpu_records_df["CPUUsage"]

        # Find peak CPU usage
        peak_idx = cpu_usage.idxmax()
        peak_time = time.iloc[peak_idx]
        peak_value = cpu_usage.iloc[peak_idx]

        mean_value = cpu_usage.mean()

        # Create interactive plot
        fig = go.Figure()

        # CPU usage line
        fig.add_trace(go.Scatter(
            x=time,
            y=cpu_usage,
            mode='lines+markers',
            fill='tozeroy',            # adds shadow under the line
            fillcolor='rgba(31,119,180,0.2)',  # semi-transparent fill
            name='Delta CPU Usage (% - 100% = 1 Core)',
            line=dict(color="#1B4EAC", width=1.5),
            marker=dict(size=2),  
            hovertemplate='Time: %{x}<br>CPU: %{y:.1f}%'

        ))

        # Highlight peak
        fig.add_trace(go.Scatter(
            x=[peak_time],
            y=[peak_value],
            mode='markers+text',
            name='Peak CPU',
            marker=dict(color='red', size=12, symbol='triangle-up'),
            text=["Peak CPU"],
            textposition='top center',
            hovertemplate='Peak CPU at %{x}: %{y:.1f}%'
        ))

        # Add average line
        fig.add_hline(y=mean_value, line_dash="3px,3px",                     
                        line_color="rgba(200, 0, 0, 0.8)",     
                        line_width=1.5,
                        annotation_text=f"Mean usage: {mean_value:.2f}%",
                        annotation_position="top left",
                        annotation=dict(
                        font=dict(color="rgba(200, 0, 0, 0.8)"), 
                        bgcolor="rgba(255,255,255,0.8)",  
                       
                        borderpad=4,
                        borderwidth=1,
                        showarrow=False
                    ))
        

        # Layout
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Delta PU Usage (%)',
            hovermode='x unified',
            autosize=True,
            plot_bgcolor="rgba(31,119,180,0.1)",
            template='plotly',  # matches VSCode dark theme
            margin=dict(l=130, r=10, t=50, b=30),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)')
            
        )
        fig.update_layout(
            title={
                'text': "Delta CPU Usage While Predicting",
                'x': 0.5,  # Center horizontally
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        return fig

    def _display_cm_rel_plot(self, return_img=True):
        fig, ax = plt.subplots()

        ax.set_title("Confusion Matrix (Normalize Per True Label)")

        normalize_per_label = self.cm.astype(float) / self.cm.sum(axis=1, keepdims=True) * 100 

        cm_disp = ConfusionMatrixDisplay(confusion_matrix=normalize_per_label, 
                                            display_labels=[AnnotationLabels.id2label[i] for i in self.cm_display_labels]
                                            )

        cm_disp.plot(ax=ax, cmap="Blues", values_format=".1f", colorbar=False)
        
        if not return_img:
            plt.plot()
        else:
            cm_disp.plot(cmap="Blues", values_format=".1f", colorbar=False, ax=ax);

            buf = io.BytesIO()

            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)

            buf.seek(0)

            # Why is there plt.close() twice you ask, well me too. only way it works though otherwise pyjupter craete additional empty plot in cell
            plt.close()
            cm_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return cm_base64

    def _display_metrics(self):

        table_style = [
            {'selector': 'caption',
            'props': [('caption-side', 'top'),
                    ('font-size', '18px'),
                    ('font-weight', 'bold'),
                    ('padding', '10px 0')]},
            {'selector': 'th',
            'props': [('background-color', '#2c3e50'),
                    ('color', 'white'),
                    ('font-size', '14px'),
                    ('text-align', 'center'),
                    ('padding', '8px 16px')]},   # smaller padding
            {'selector': 'td',
            'props': [('font-size', '13px'),
                    ('text-align', 'center'),
                    ('padding', '12px 18px')]},   # smaller padding
            {'selector': 'table',
            'props': [('border-collapse', 'separate'),
                    ('border-spacing', '0 6px'),
                    ('table-layout', 'fixed'),  # <-- fixed layout
                    ('width', '100%'),
                    ('margin-left', '10px'),
                    ('margin-right', '10px')]}          # <-- always fill cell
        ]
        

        ###### Metric per label
        metric_per_label_df = pd.DataFrame(self.metric_per_label).T

        metric_per_label_df_styled = metric_per_label_df.style.set_caption("Metrics per Label") \
            .set_table_styles(table_style).format({
                "precision": "{:,.5f}",
                "recall": "{:,.5f}",    
                "f1-score": "{:,.5f}",       
                "support": "{:,.0f}"        
            })


        # display(metric_per_label_df_styled)

        ###### Accuracy
        pd.DataFrame([{"Accuracy": self.accuracy}]).T

        accuracy_df = pd.DataFrame([{"Accuracy": self.accuracy}]).T
        accuracy_df_styled = accuracy_df.style.hide(axis=1).set_caption("Accuracy") \
            .set_table_styles(table_style).format({
                "precision": "{:,.5f}",
                "recall": "{:,.5f}",    
                "f1-score": "{:,.5f}",       
                "support": "{:,.0f}"        
            })

        # display(accuracy_df_styled)

        ###### weighted and macro average
        weighted_and_macro_avg_df = pd.DataFrame({"Weighted Average": self.weighted_avg, "Macro Average": self.macro_avg}).T
        weighted_and_macro_avg_df_styled = weighted_and_macro_avg_df.style.set_caption("Weighted and Macro Avg") \
            .set_table_styles(table_style).format({
                "precision": "{:,.5f}",
                "recall": "{:,.5f}",    
                "f1-score": "{:,.5f}",       
                "support": "{:,.0f}"        
            })

        # display(weighted_and_macro_avg_df_styled)
        ###### run time df
        runtime_df = pd.DataFrame([{"Runtime (seconds)": self.model_runtime, "Avg time per prediction (seconds)": self.avg_time_per_prediction}]).T
        runtime_df_styled = runtime_df.style.hide(axis=1).set_caption("Runtime Stats") \
            .set_table_styles(table_style).format({
                "precision": "{:,.5f}",
                "recall": "{:,.5f}",    
                "f1-score": "{:,.5f}",       
                "support": "{:,.0f}"        
            })
        
        return metric_per_label_df_styled,  weighted_and_macro_avg_df_styled, accuracy_df_styled , runtime_df_styled

    def _display_mem_usage_plot(self):
        
        mem_records_df = pd.DataFrame(list(self.memory_records.items()), columns=["TimeDelta", "MemUsage"])
        # only get mem usage delta
        mem_usage_t0 = mem_records_df[mem_records_df["TimeDelta"] == 0]["MemUsage"][0]
        mem_records_df["MemUsageDelta"] = mem_records_df["MemUsage"] - mem_usage_t0
        # convert mem usage to MB
        mem_records_df["MemUsageDeltaMB"] = mem_records_df["MemUsageDelta"] / (1024 * 1024)

        time = mem_records_df["TimeDelta"]
        mem_usage = mem_records_df["MemUsageDeltaMB"]

        # Find peak Mem usage
        peak_idx = mem_usage.idxmax()
        peak_time = time.iloc[peak_idx]
        peak_value = mem_usage.iloc[peak_idx]

        mean_value = mem_usage.mean()

        # Create interactive plot
        fig = go.Figure()

        # CPU usage line
        fig.add_trace(go.Scatter(
            x=time,
            y=mem_usage,
            mode='lines+markers',
            fill='tozeroy',            # adds shadow under the line
            fillcolor='rgba(31,119,180,0.2)',  # semi-transparent fill
            name='Delta Mem Usage MB',
            line=dict(color="#2C51E1", width=1.5),
            marker=dict(size=2),
            hovertemplate='Time: %{x}<br>CPU: %{y:.1f}%'
        ))

        # Highlight peak
        fig.add_trace(go.Scatter(
            x=[peak_time],
            y=[peak_value],
            mode='markers+text',
            name='Peak Mem',
            marker=dict(color='red', size=12, symbol='triangle-up'),
            text=["Peak Mem"],
            textposition='top center',
            hovertemplate='Peak Mem at %{x}: %{y:.1f}%'
        ))


        
        fig.add_hline(y=mean_value, line_dash="3px,3px",                     
                line_color="rgba(200, 0, 0, 0.8)",     
                line_width=1.5,
                annotation_text=f"Mean usage: {mean_value:.2f} MB",
                annotation_position="top left",
                annotation=dict(
                font=dict(color="rgba(200, 0, 0, 0.75)"), 
                bgcolor="rgba(255,255,255,0.8)",  

                borderpad=4,
                borderwidth=1,
                showarrow=False
            ))

        # Layout
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Delta Mem Usage MB',
            hovermode='x unified',
            autosize=True,
            plot_bgcolor="rgba(31,119,180,0.1)",
            template='plotly',  
            margin=dict(l=130, r=10, t=50, b=30),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)')
            
        )
        fig.update_layout(
            title={
                'text': "Delta Mem Usage While Predicting",
                'x': 0.5,  # Center horizontally
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        return fig

    def _show_report_shell(self):
        print(f"#####\t Report for Model: {self.model_title}\t\n")

        print("Metrics per label\n")
        print(pd.DataFrame(self.metric_per_label))

        print(f"\nAccuracy: {self.accuracy}\n")

        print("\nMacro Average\n")
        print(pd.DataFrame([self.macro_avg]).to_string(index=False))


        print("\nWeighted Average\n")
        print(pd.DataFrame([self.weighted_avg]).to_string(index=False))
    
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

    def show_report(self):
        if self._is_in_jupyter_notebook():
            self._show_report_jupiter()
        else:
            self._show_report_shell()
   
    def _is_in_jupyter_notebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or JupyterLab
            else:
                return False  # Other environments (IPython terminal, standard Python)
        except NameError:
            return False       # Standard Python interpreter

    def _show_report_jupiter(self):
        
        display(HTML("""
        <style>
        div.center_div {
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            width: 100%;
        }
        </style>
        """))


        HTML_title, HTML_subtitle = self._display_titles()
        metric_per_label_df_styled,  weighted_and_macro_avg_df_styled, accuracy_df_styled, runtime_df_styled = self._display_metrics()

        cm_rel = self._display_cm_rel_plot()
        cm_abs = self._display_cm_abs_plot()

        cpu_fig = self._display_cpu_usage_plot()
        cpu_plot = cpu_fig.to_html(include_plotlyjs='cdn', full_html=False)

        mem_fig = self._display_mem_usage_plot()
        mem_plot = mem_fig.to_html(include_plotlyjs='cdn', full_html=False)

        # --- Combine in HTML grid ---
        html = f"""
            {HTML_title}
        <br/>
            {HTML_subtitle}
            <br/>
            <br/>
        <div class="center_div">
            <div style="display: grid; grid-template-columns: 470px 470px; grid-gap: 10px;">
                <div style="padding:5px; overflow-x:auto; width:100%;">{metric_per_label_df_styled.to_html()}</div>
                <div style="padding:5px; overflow-x:auto; width:100%;">{weighted_and_macro_avg_df_styled.to_html()}</div>
                <div style="padding:5px; width:100%;">{accuracy_df_styled.to_html()}</div>

                <div style="padding:5px; width:100%;">{runtime_df_styled.to_html()}</div>
                                
                <div style="padding:5px;"><img src="data:image/png;base64,{cm_abs}" style="width:100%"></div>
                <div style="padding:5px;"><img src="data:image/png;base64,{cm_rel}" style="width:100%"></div>
                <div style="grid-column: span 2;">{mem_plot}</div>
                <div style="grid-column: span 2;">{cpu_plot}</div>
            </div>
        </div>
        """

        display(HTML(html))

    def save_to_file(self, path):
        with open(path, "wb") as f:
           pickle.dump(self, f)

    def load_report(path):
        with open(path, "rb") as f:
            return pickle.load(f)

class ReportsComparison(object):
    def __init__(self, reports):
        self._reports = reports

    def _get_report_dict(report):
        report_dict = {"Model" : report.model_title, 
               "Accuracy": report.accuracy,
               "Average Time Per Prediction": report.avg_time_per_prediction
        }

        for macro_avg_item in report.macro_avg.keys():
            report_dict[f"Macro Average ({macro_avg_item})"] =  report.macro_avg[macro_avg_item]

        for weighted_avg_item in report.weighted_avg.keys():
            report_dict[f"Weighted Average ({weighted_avg_item})"] =  report.weighted_avg[weighted_avg_item]

        return report_dict

    def get_comparison_df(self):
        report_dicts = []
        for report in self._reports:
            report_dicts.append(ReportsComparison._get_report_dict(report))
        return pd.DataFrame(report_dicts)
    
    def compare_and_show(self):
        df = self.get_comparison_df()

        # Choose default metrics to show
        default_metrics = ["Accuracy", "Average Time Per Prediction", "Weighted Average (precision)"]

        # Initialize Dash app
        app = Dash(__name__)

        # Layout with styled checklist and default selected columns
        app.layout = html.Div([
            html.Div([
                html.H4(
                    "Select Metrics to Display:",
                    style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '8px'}
                ),
                dcc.Checklist(
                    id='column-selector',
                    options=[{"label": col, "value": col} for col in df.columns if col not in ["Model"]],
                    value=default_metrics,
                    inline=False,  # disable inline to allow wrapping
                    inputStyle={"transform": "scale(1.5)", "margin-right": "6px", "margin-left": "10px"},
                    labelStyle={
                        "margin": "5px",
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '14px',
                        'fontWeight': 'bold',
                        'color': '#333',
                        'display': 'inline-block',
                        'width': '150px'  # adjust width to control number per row
                    },
                    style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'left'}
                )
            ], style={
                'textAlign': 'center',
                'padding': '10px',
                'backgroundColor': '#f9f9f9',
                'borderRadius': '8px',
                'marginBottom': '15px',
                'display': 'inline-block'
            }),
            
            dash_table.DataTable(
                id='model-metrics-table',
                data=df.to_dict('records'),
                sort_action="native",
                sort_mode="single",
                style_table={'width': 'auto', 'height': '400px', 'overflowX': 'auto', 'overflowY': 'auto', 'margin': 'auto'},
                style_cell={
                    'textAlign': 'center',
                    'padding': '4px',
                    'minWidth': '40px',
                    'width': 'auto',
                    'maxWidth': '150px',
                    'whiteSpace': 'normal',
                    'fontSize': '13px',
                    'lineHeight': '15px',
                    'fontFamily': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': '#f0f0f0',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'fontSize': '14px',
                    'height': '30px',
                    'fontFamily': 'Arial, sans-serif'
                },
                style_header_conditional=[]
            )
        ])

        # Callback to update table columns dynamically and highlight sorted header
        @app.callback(
            Output('model-metrics-table', 'columns'),
            Output('model-metrics-table', 'style_header_conditional'),
            Input('model-metrics-table', 'sort_by'),
            Input('column-selector', 'value')
        )
        def update_table_columns(sort_by, selected_metrics):
            displayed_columns = ["Model"] + selected_metrics
            new_columns = []
            style_header_conditional = []

            numeric_cols = df.select_dtypes(include='number').columns


            for col in displayed_columns:
                col_name = col
                
                col_dict = {"name": col_name, "id": col}

                # Add numeric formatting if column is numeric
                if col in numeric_cols:
                    col_dict["type"] = "numeric"
                    col_dict["format"] = {'specifier': '.5f'}

                if sort_by and col == sort_by[0]['column_id']:
                    direction = sort_by[0]['direction']
                    col_name = f"{col} ({direction})"
                    color = '#d1e7dd' if direction == 'asc' else '#f8d7da'
                    style_header_conditional.append({
                        'if': {'column_id': col},
                        'backgroundColor': color
                    })
                new_columns.append(col_dict)

            return new_columns, style_header_conditional

        # Run inline in Jupyter
        app.run(mode='inline')

    


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
        report.model_info = self.model.model_info
        
        y_true = self.eval_dataset["label"]

        y_pred, report = self._predict_and_profile(report=report)

        metrics_dict = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)

        report.add_classfication_report_dict(metrics_dict)

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
    def __init__(self, model, title, model_info=None):
        """
        :param model: Model to evaluate, the model should be already trained and implement the methods in  
        :param title: Title or label for this evaluation instance
        :param model_info: more infomation abotu the model 
        """
        assert isinstance(model, ModelEvalWrapperInterface), "model must implement the ModelEvalWrapperInterface Protocol"
        self.model = model
        self.title = title
        self.model_info = model_info

