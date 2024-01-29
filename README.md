<h1 align="center"> <p> Dual Operating Modes of In-Conctext Learning </p></h1>
<h4 align="center">
    <p><a href="https://myhakureimu.github.io/" target="_blank">Ziqian Lin</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>UW-Madison</p>
    </h4>

**Paper Link**: TBD link

In-context learning (ICL) exhibits dual operating modes: \emph{task learning}, \ie{} acquiring a new skill from in-context samples, and \emph{task retrieval}, \ie{}, locating and activating a relevant pretrained skill.
Recent theoretical work investigates various mathematical models to analyze ICL, but existing models explain only one operating mode at a time.
We introduce a probabilistic model, with which one can explain the dual operating modes of ICL simultaneously.
Focusing on in-context learning of linear functions, we extend existing models for pretraining data by introducing multiple task groups and task-dependent input distributions. 
We then analyze the behavior of the optimally pretrained model under the squared loss, \ie{}, the MMSE estimator of the label given in-context examples.
Regarding pretraining task distribution as prior and in-context examples as observation, we derive the closed-form expression of the task posterior distribution. 
With the closed-form expression, we obtain a quantitative understanding of the two operating modes of ICL.
Furthermore, we shed light on an unexplained phenomenon observed in practice: under certain settings, the ICL risk initially increases and then decreases with more in-context examples.
Our model offers a plausible explanation for this ``early ascent'' phenomenon: a limited number of in-context samples may lead to the retrieval of an incorrect skill, thereby increasing the risk, which will eventually diminish as task learning takes effect with more in-context samples.
We also theoretically analyze ICL with incorrect labels, \eg, zero-shot ICL, where in-context examples are assigned random labels.
Lastly, we validate our findings and predictions via experiments involving Transformers and large language models.

# Experiments
The following sections give guidance for reproducing all the experiments in the paper.
## Numerical Computation
### Figure 4
```bash
cd NumericalComputation/Figure4/
python BayesianSimulation_prediction.py
```
### Figure 5
```bash
cd NumericalComputation/Figure5/
python 5.1.2.py
```
### Figure 6
```bash
cd NumericalComputation/Figure5/
python 5.1.3-EarlyAscentWithCorrectLabels.py
```
## RealWorld LLM Experiment
### Table 1
```bash
cd RealWorldLLMExperiment/Table1/
vi call_openai.py
```
replace "yourkey" in the file with your openai key
```bash
python Ushape.py
```
### Figure 8
Note: In the following codes, the inferences of llama2, mistral, and mixtral are based on [vllm](https://docs.vllm.ai/en/latest/), you will need at least 4xA100 to run the biggest models including mixtral and llama-2-70b-hf.
```bash
cd RealWorldLLMExperiment/Figure8/
vi call_openai.py
```
Replace "yourkey" in the file with your openai key
```bash
python test_gpt4.py
python test_llama-2-13b-hf.py
python test_llama-2-70b-hf.py
python test_mistral.py
python test_mixtral.py
```
After finishing running the five experiments, run:
```
python 5.2.2-ZeroICL.py
```
## Transformer Experiment
The following code can be run on a single 4090 GPU.
### Figure 9
```bash
python TS_Regular4_4_run.py
python TS_Regular4_delta_visual.py
```
### Figure 10
```bash
python TS_RegularM_run.py
python TS_RegularM_visual.py
```
### Figure 11
```bash
python TS_D_d_run.py
python TS_D_d_visual.py
```
