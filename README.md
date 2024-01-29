<h1 align="center"> <p> Dual Operating Modes of In-Conctext Learning </p></h1>
<h4 align="center">
    <p><a href="https://myhakureimu.github.io/" target="_blank">Ziqian Lin</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>UW-Madison</p>
    </h4>

**Paper Link**: TBD link

TBD abstract

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
Note: In these codes, the inferences of llama2, mistral, and mixtral are based on [vllm](https://docs.vllm.ai/en/latest/), you will need at least 4xA100 to run the biggest models including mixtral and llama-2-70b-hf.
## Transformer Experiment
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
