# A cortical column inspired system

This project is investigating the properties of homogeneous distributed neural networks in conjunction with movement (as opposed to NCAs). Therefore, I have created a system inspired by the columnar neocortex, and use the whisker barrel cortex as close inspiration. It has an architecture where identical modules process their neighborhood to classify objects. They can also move about in the sensor space, and process neighborhoods there, as seen in the figure below.  

It is applied to classification, because of the connection of such a system to cortical columns and Jeff Hawkins' idea of distributed classification and voting. With some reworking, the system can be translated to an NCA like ones typically used for robot control with message passing. 

![A cute rat](img/project_description.png)

The fully formed system will be tested for various forms of robustness, scalability, and adaptability. I also want to test different forms of teaching the system, like input information and specific loss.

The system can be used for 
1. Investigating properties of homogeneous distributed neural network systems with and without movement for engineering
2. ... Or for theorizing about neocortex?
3. Potentially zero-shot scalable robot bodies for classification

It's still in progress.

## Code structure

The configuration for experiments are given in "config" files. To run an experiment with the current config, run 

```python
python3 main.py
```

There are several flags to modify the output of main.py, f.ex. "-s" will allow you to save. With this option, a folder "experiments" will be made in the folder, and a sub-folder will be added that contains your results. All sub-folder names are unique. 

To plot the results from any result folder, run 

```python
python3 plot_runs.py path/to/your/experiment/folder
```

Path can be absolute or relative. 

The folder "src" contains the functionality used in main.py. Most notably, "moving_nca.py" contains the class of the new system. But keep in mind that how to train the system lies in main.py. 

The script zero_shot_scalability.py is currently outdated. TODO. 

## Authors

Mia-Katrin Kvalsund, Kai Olav Ellefsen, Kyrre Glette, Sidney Pontes-Filho, Mikkel Elle Lepperød

Code mainly produced by: Mia-Katrin Kvalsund

Some of Sidney Pontes-Filho's code is also used from this project: [Neural Cellular Robot Substrate](https://github.com/sidneyp/neural-cellular-robot-substrate)



