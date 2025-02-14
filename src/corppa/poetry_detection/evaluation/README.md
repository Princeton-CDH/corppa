# Evaluation
This directory contains code to evaluate methods for detecting (and identifying)
poetry excerpts.

## Terminology
In this setting, we assume we have two sets of annotations: a *reference* set and
a *system* set. In the typical scenario, we are evaluating some system's results
against some baseline annotations serving as a sort of "ground truth". This lets
us ask how well the system identifies poetry excerpts compared to this baseline
(e.g., how many does it miss? how many unexpected excerpts does it flag?). That
said, we can use the same code to compare the agreement of two sets of annotations
(produced by humans or computers).

## Evaluation Dataset
In order to evaluate computational methods for detecting poetry excerpts, we
built a small evaluation dataset to use as our reference set. For a small subset of
adjudicated spans from our annotation task, we manually identified the poems and
constructed a `.JSONL` file containing these annotations to use for evaluation.

## Page-level Span Evaluation Method
We compare the poetry span annotations produced at the page level calculating page-level
precision and recall scores. Our approach builds off existing methods for evaluation span
annotations, but makes design choices suitable to our task.

### Assumptions
**Spans.** In this setting a span to have the following three components:
- start: Start index of the span
- end: End index of the span
- label: The ID of the referenced poem

Note that a span's interval is Pythonic [closed, open) interval.

**Reference Spans.** We assume that a reference's spans do not overlap. Either the entire span
is an excerpt from a poem or it is not. Note that our method can generally accomodate overlapping
spans if they have different labels (i.e., one poem reuses a line from yet another poem), but only
if poem labels are taken into account.

**System Spans.** Unlike reference spans, system spans are permitted to overlap. This can easily
happen in the case of passim where overlapping passages can be identified for different lines of
the same reference poem.

## Span Overlap Factor
Taking inspiration from Stanislav Olenchenko's [blog post](https://blog.p1k.org/yet-another-way-of-ner-systems-evaluation/),
we define the overlap factor between two spans $a$ and $b$ *with the same poem label* to be equal to
the proportion of the overlap of their intervals with respect to the larger span.

$$
\text{overlap-factor}(a, b) = \frac{\min(a_{end}, b_{end}) - \max(a_{start}, b_{start})}{\max(a_{end}-a_{start}, b_{end}-b_{start})}
$$

### Step 1. Map reference spans to viable system spans
In this first step, we take inspiration from ACE NER evaluation metrics (see Hal Daume's [blog post comment](https://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html?showComment=1156981200000#c115698122985619877)). We effectively build a bipartite graph linking
each reference span to the "best" matching system span. Currently, we determine the best matching
system span for each reference span independently. For each reference span, we select the
system span with the greatest overlap factor. This may result multiple reference spans linking to
the same system span.

Note that this selection process will greatly penalize systems that separate a poem excerpt into
multiple spans.

### Step 2. Determine Reference-"System" Span Pairs
To calculate precision and recall scores, we must have one-to-one style mappings between reference
and system spans. This is straightforward from our reference-system span mapping if each system
span is linked with at most one reference span. However, if $k$ reference spans map to a
single system span, we'll need to split the system span into $k$ distinct subspans.

#### Splitting System Spans
For reference spans $r_0 \dots r_k$ mapping to a single system span $s$, we split $s$ into the following
$k$ sub-spans using of the starting index of all but the first reference span:

$$
\left(s_{start}, r_{1_{start}} \right),
\left(r_{1_{start}}, r_{2_{start}} \right),
\cdots,
\left(r_{k_{start}}, s_{end} \right)
$$


### Step 3. Calculating Precision and Recall


## Script Details
