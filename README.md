<h1 align="center"> <p> Dual Operating Modes of In-Conctext Learning </p></h1>
<h4 align="center">
    <p><a href="https://myhakureimu.github.io/" target="_blank">Ziqian Lin</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>UW-Madison</p>
    </h4>

**Paper Link**: [arxiv.org/abs/2402.18819](https://arxiv.org/abs/2402.18819)

In-context learning (ICL) exhibits dual operating modes: **task learning**, i.e., acquiring a new skill from in-context samples, and **task retrieval**, i.e., locating and activating a relevant pretrained skill.
Recent theoretical work investigates various mathematical models to analyze ICL, but existing models explain only one operating mode at a time.
We introduce a probabilistic model, with which one can explain the dual operating modes of ICL simultaneously.
Focusing on in-context learning of linear functions, we extend existing models for pretraining data by introducing multiple task groups and task-dependent input distributions. 
We then analyze the behavior of the optimally pretrained model under the squared loss, i.e., the MMSE estimator of the label given in-context examples.
Regarding pretraining task distribution as prior and in-context examples as observation, we derive the closed-form expression of the task posterior distribution. 
With the closed-form expression, we obtain a quantitative understanding of the two operating modes of ICL.
Furthermore, we shed light on an unexplained phenomenon observed in practice: under certain settings, the ICL risk initially increases and then decreases with more in-context examples.
Our model offers a plausible explanation for this "early ascent" phenomenon: a limited number of in-context samples may lead to the retrieval of an incorrect skill, thereby increasing the risk, which will eventually diminish as task learning takes effect with more in-context samples.
We also theoretically analyze ICL with incorrect labels, e.g., zero-shot ICL, where in-context examples are assigned random labels.
Lastly, we validate our findings and predictions via experiments involving Transformers and large language models.

# Experiments
The following sections give guidance for reproducing all the experiments in the paper.

## Accessing Data for Efficient Experiment Replication 
To replicate the experiments efficiently, download the .zip files from the provided [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0) and unzip them directly into the corresponding directory within your cloned or downloaded GitHub repository. (You need a Dropbox account first. Register one with any email for free! Do not spend money on it to download the data!) For instance, if the .zip file resides in "NumericalComputation/Figure4/" within Dropbox, it should be unzipped to "NumericalComputation/Figure4/" in your local repository. Please note that some experimental outcomes are not included in this link due to their execution time.

## Setup for Non Real-world LLM
### System
Ubuntu 22.04.3 LTS

Python 3.10.12
### Package
setproctitle              1.3.2

matplotlib                3.7.2

tqdm                      4.66.1

scikit-learn              1.3.2

scipy                     1.11.2

pytorch                   2.0.1

## Numerical Computation
### Figure 4
#### Step 1: Go to the Folder
```bash
cd NumericalComputation/Figure4/
```
#### Step 2 (Method 1): Get Results from Scratch
```bash
python BayesianSimulation_Preprocess.py
```
One can reduce the sample size "K = 20000" for the Monte Carlo simulation in the code to accelerate the process, though this will likely result in increased variance.
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
Then run
```bash
python BayesianSimulation_Visualize.py
```
to get Figure4.pdf.

### Figure 5
#### Step 1: Go to the Folder
```bash
cd NumericalComputation/Figure5/
```
#### Step 2 (Method 1): Get Results from Scratch
```bash
python 5.1.2_Preprocess.py
```
One can reduce the sample size "K = 80000" for the Monte Carlo simulation in the code to accelerate the process, though this will likely result in increased variance.
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
Then run
```bash
python 5.1.2_Visualize.py
```
to get Figure5.pdf.

### Figure 6
#### Step 1: Go to the Folder
```bash
cd NumericalComputation/Figure5/
```
#### Step 2 (Method 1): Get Results from Scratch
```bash
python EarlyAscent_Preprocess.py
```
One can reduce the sample size "K = 10000" for the Monte Carlo simulation in the code to accelerate the process, though this will likely result in increased variance.
**Note**: The code takes a long time to run since it loops through these parameters:
d_list = \[1,2,3,5,8\]
and
demon_list = \[0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072\].

#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
Then run
```bash
python EarlyAscent_Visualize.py
```
to get Figure6.pdf.

## RealWorld LLM Experiment
### Table 1 (running with GPT-4 2023/11/20)
#### Step 1: Go to the Folder
```bash
cd RealWorldLLMExperiment/Table1/
```
#### Step 2: Register Your Openai key
```bash
vi call_openai.py
```
Replace the string "yourkey" in the code with your Openai key.
#### Step 3: Get Results from Scratch
For k (for instance k=4) in-context examples, run
```bash
python Ushape.py --k 4
```

### Figure 8
Note: In the following codes, the inferences of llama2, mistral, and mixtral are based on [vllm](https://docs.vllm.ai/en/latest/). One will need at least 4xA100 to run the biggest models, including mixtral and llama-2-70b-hf.
#### Step 1: Go to the Folder
```bash
cd RealWorldLLMExperiment/Figure8/
```
#### Step 2: Register Your Openai key
```bash
vi call_openai.py
```
Replace the string "yourkey" in the code with your Openai key.
#### Step 3 (Method 1): Get Results from Scratch
```bash
python test_gpt4.py
python test_llama-2-13b-hf.py
python test_llama-2-70b-hf.py
python test_mistral.py
python test_mixtral.py
```
#### Step 3 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 4: Visualize Results
After step 3, run:
```
python ZeroICL.py
```

## Transformer Experiment
The following code can be run on a single 4090 GPU.
#### Step 1: Go to the Folder
```bash
cd TransformerExperiment/
```
### Figure 9
#### Step 2 (Method 1): Get Results from Scratch
```bash
python TS_Regular4_delta_run.py
```
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
```bash
python TS_Regular4_delta_visual.py
```
### Figure 10
#### Step 2 (Method 1): Get Results from Scratch
```bash
python TS_RegularM_run.py
```
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
```bash
python TS_RegularM_visual.py
```
### Figure 11
#### Step 2 (Method 1): Get Results from Scratch
```bash
python TS_D_d_run.py
```
#### Step 2 (Method 2): Download Results from Dropbox
Download and unzip the corresponding .zip file from [Dropbox link](https://www.dropbox.com/scl/fo/q0rj5eyfd9wasatbnpy7r/h?rlkey=epjq87hvf3br3ljqa6a1g50bn&dl=0).
#### Step 3: Visualize Results
```bash
python TS_D_d_visual.py
```
