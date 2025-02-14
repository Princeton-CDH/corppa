# Evaluation
This directory contains code to evaluate methods for detecting and identifying
poetry excerpts.


## Terminoloy
In this setting, we assume we have two sets of annotations: a *reference* set and
a *system* set. In the typical scenario, we are evaluating some system's results
against some baseline annotations serving as a sort of "ground truth". This lets
us ask how well the system identifies poetry excerpts compared to this baseline
(e.g., how many does it miss? how many unexpected excerpts does it flag?). That
said, we can use the same code to compare the agreement of two sets of annotations
(produced by humans or computers).

In our particular setting, our evaluation dataset is a small subset of PPA pages
from our adjudicated dataset. These hand-identified spans of poetry were then
manually identified to particular poems.

## Page-level Span Evaluation
TODO
