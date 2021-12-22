# Towards an Event Based Plot Model (JCLS 2022)

## Overview and ressources

This repository provides annotation data, analysis results and visualization results for our contribution to JCLS2022.
Specifically, there are the following resources:
- `./event_annotations_with_summary_data/`: event annotations for our four manually annotated texts
 icluding the freqeuncy of summaries that mention a specific event
- `narrpy` a python package used for the provided visualizations
- `./summary_annotations/`: the manual summary annotations
- `./summary_optimization_results/`: a json file which provides the data our optimization results are based on
- `jcls2022.ipynb`: a Jupyter Notebook that provides the visulizations of our paper in an interactive version

## Prerequisites

`narrpy`and the Jupyter Notebook have the following requirements:
- numpy==1.21.0
- seaborn==0.11.2
- matplotlib==3.4.3
- matplotlib-inline==0.1.3
- pandas==1.3.4
- plotly==5.3.1