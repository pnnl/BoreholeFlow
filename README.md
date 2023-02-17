# BoreholeFlow Repository

This repository contains Python functions to perform analysis and scripts to create figures in the paper:

Day-Lewis, F.D., Mackley, R.D., and Thompson, J., Interpreting Concentrations Sampled from Long-Screened Wells in the Presence of Borehole Flow: An Inverse Modeling Approach

## Dependencies:

To install quad prog: 

```pip install quadprog``` 

https://pypi.org/project/quadprog/

## Contents:

### Class BhFlow:

Functions:
- ***NumberOfLayers***: Creates a variable defining the number of layers based on the number of K values given
- ***NlayersZerosArray, NxNLayersZerosArray, NlayersOnesArray***: These three functions create arrays that depend on the number of layers
- ***Discretize***: Discretize the screened portion of the well based on interval thickness (b) and depth to bottom of well
- ***BoreholeFlowModel***: Solves steady state hydraulic flow problem
- ***BoreholeTransportModel***: Solves steady state transport
- ***TransientTransportModel***: Solves transient transport
- ***invertBHconc***: Invert concentration data
- ***MakeDataLists***: These functions are for data cleaning (combining datasets, converting to arrays, etc.)
- ***AppendData***
- ***MakeDataListsTransient***
- ***AppendDataTransient***
- ***Convert2Arrays***
- ***Convert2ArraysTransient***
- ***CombineFandTOutput***

### Class PlotFigures:

Plots figures 3, 4, 5, and 6

### Scripts:

- ***BoreholeFlow.py***: library of classes/functions (detailed above) to be used in the Jupyter Notebook scripts.
- ***Borehole_Algorithm_Data_Validation.ipynb***: Imports output data from Matlab and Python for comparison. The variables compared each showcase a different function's output and validate the match between results.
- ***makeFigure3.ipynb***: makes figure 3 in the paper
- ***makeFigure4.ipynb***: makes figure 4 in the paper
- ***makeFigure5.ipynb***: makes figure 5 in the paper
- ***makeFigure6.ipynb***: makes figure 6 in the paper

### pdf:

This 'BoreholeFlow Notebooks.pdf' has the Jupyter notebooks to facilitate viewing.

## License:

This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830

## Open source license (BSD-style):

BoreholeFLow
Copyright © 2023, Battelle Memorial Institute
All rights reserved.

1. Battelle Memorial Institute (hereinafter Battelle) hereby grants 
permission to any person or entity lawfully obtaining a copy of this software 
and associated documentation files (hereinafter “the Software”) to redistribute 
and use the Software in source and binary forms, with or without modification.  
Such person or entity may use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and may permit others to do so, subject to the 
following conditions:

- Redistributions of source code must retain the above copyright notice, this list 
of conditions and the following disclaimers. 
- Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution. 
- Other than as used herein, neither the name Battelle Memorial Institute or Battelle 
may be used in any form whatsoever without the express written consent of Battelle.  

2. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

