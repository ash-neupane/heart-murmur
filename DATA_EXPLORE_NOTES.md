## Dataset Exploration
After inspecting the files, it seems training_data.csv has the metadata loaded. Could be useful to investigate general characterstics of the dataset. The dataset looks very clean outside of a few missing values in age, height, weight columns.

**Things that are likely to be label columns:**
- Outcome: approx. even split between Normal and Abnormal
- Murmur: Imbalanced, mostly absent, some unknown too. The 119 data points labeled "unknown" did not meet the required signal quality standards according to the dataset description.

**More granular Murmur annotations for cases with Murmur:**

Seems like the murmur location data is available for all known murmurs? The location annotations are semi-supervised, but a human annotator went through all the annotations and made sure it's good.
- Murmur locations: indicates which recording locations it's present in
- Most audible location: where the murmur is most audible.
- Systolic Murmur timing, shape, grading, pitch, quality. Available for most true labels [178/179 murmurs]
- Diastolic Murmur timing, shape, grading, pitch, quality. More sparse [5/179 murmurs]

**Things that are likely to be feature columns:**
- Recording Location: seems to be a combo of AV, PV, TV, MV, most of the data has all 4, a significant portion is a subset. There's a few with duplicated AV+AV or MV+MV -> I wonder if they are poor quality data that needs to be filtered out?

- Age: Imbalanced. Mostly children and exponential drop off.

- Sex: Seems balanced male and female

- Pregnancy Status: Only applies to ~half the population but still imbalanced `<10%` True

- Height: avg 110 (cm), std: 30, min 35, max 180. Probably normally distributed. Nothing noteworthy as of now, at least nothing you wouldn't expect knowing the age ranges from infants to adolescents.

- Weight: avg 23.6 (kg), std 15, min 2.3, max 110.8.

**IDs:**
- Patient ID: primary id
- Additional ID: links patients who were in both campaigns

## Focus on the first part of the problem
These metadata will come in handy for the open ended part of the problem, but the first part of the problem is to simply load the data.

However, the type of ML model being trained is important. Is it supervised or unsupervised? For unsupervised case, we may want to keep some high quality labeled data separate for evaluation. For supervised, we need to figure out which data-label to return - the unlabeled data should probably be ignored.

*We would like to distinguish systolic murmurs (that occur when the heart's ventricles are filling with blood) from diastolic murmurs (that occur when the blood is being ejected from the ventricles into systemic circulation). You should build a dataloader for training a machine learning model in pytorch to solve this problem.*

But there are only 5 rows in this csv file with diastolic murmur annotations. If that is actually what separates systolic from diastolic murmurs, then we have a big imbalance of 178 vs. 5. 5 is generally a very very small number of labels to have, and the imbalance isnt ideal.

**Biology Context**

More turbulent the blood flow, the louder the signal.

Systole: Heart contracts and pumps blood into the body
Diastole: Heart relaxes and refills with blood

AV: Aortic Valve, MV: Mitral Valve, PV: Pulmonic Valve, TV: Tricuspid Valve - they are locations in the chest and signals are collected from these locations by controlling where the stethescope is.

S1: MV + TV - hard to distinguish the two as they occur close together. At the beginning of systole
S2: AV + PV - at closure of these valves, beginning of diastole. AV closes earlier, so AV vs. PV is more discernible than in the case of S1.

**Signal Properties**

Sampling rate = 4000 Hz

Global file (.txt)
line 1 = [id] [n_recording_locs] [fs]
then the file names for the 4 recording locations
then the annotations and metadata, all prefixed by `#`

Header file (.hea)
line 1: [record name] [num_channels] [fs] [n_samples in recording]
line 2: [signal filename] [format=16 byte with 44 offset for wav header] [Gain] [n_bits per sample] [offset] [init value] [checksum] [block size] [signal recording location]

Audio File (.wav)

Segmentation Annotation File (.wav):
column 1: t_start. time when wave is first detected
column 2: e_end. time when wave is last detected
column 3: label:
```
{
  0: Unannotated signal,
  1: S1 Wave,
  2: Systolic Period,
  3: S2 Wave,
  4: Diastolic Period
}
```
Order goes S1 (start of systole) -> Systolic Period -> S2 (start of diastole) -> Diastolic Period. Most of the labels are cycle through the 5 options in the dictionary in order.

**What does this mean for labels?**
I can imagine using the murmur dataset and trying to figure out from this annotation if the murmur came from systolic v diastolic period. A bit unclear at the moment exactly what the label would be. Doesn't seem like a fully supervised problem just yet.

A few questions:
1. Since we are trying to detect systolic v. diastolic murmurs, should we only load the data where there are murmurs? Could the audio files where murmurs are not detected also be useful?
1. We don't explicitly have labels for where the murmur occured, but we have a global label for whether there was a murmur, and there are cases where there were both systolic and diastolic murmurs??
1. Does the problem ask us to load random samples from all available audio data?
1. From manual inspection, the audio signal lengths are anywhere from less than 15 secs to 30+ seconds, but seem to be in that order of magnitude(verify with a script?). And how do we account for the segmentation annotation? For signals longer than 15 seconds it may make sense to produce one 15 second chunk out of it which fully contained the annotated parts of the signal. How to handle the signals that are shorter? Should we discard them or pad them?
