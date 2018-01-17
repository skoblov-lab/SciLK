BioCreative IV CHEMDNER corpus (Version 1.0 30-09-2014)
-------------------------------------------------------

If you use this corpus, please cite:


Krallinger, M.  et al. The CHEMDNER corpus of chemicals and drugs and its annotation principles.
J Cheminform, 2014



This directory contains the full CHEMDNER abstracts and annotations.


1) Directory content
--------------------

BioC.dtd → the DTD for the BioC XML corpus format

chemdner.key → the .key file for the BioC infon keys and additional format information

chemdner_abs_test_pmid_label.txt → see [#5] below (evaluation set abstract classification)

cdi_ann_test_13-09-13.txt → see [#6] below (CDI subtask evaluation data set)

cem_ann_test_13-09-13.txt → see [#7] below (CEM subtask evaluation data set)

evaluation.* → evaluation (test set) files - see [#2]

development.* → development files - see [#2]

plain2bioc.py → a Python script to convert CHEMDNER plain text files into BioC XML files

Readme.txt → this file

training.* → training files - see [#2]

*.predictions.* → team run predictions


2) Corpus file formats
----------------------

*.abstracts.txt → TSV-formatted plain-text of the PubMed abstracts - see [#3]

*.annotations.txt → TSV-formatted plain-text of the annotations - see [#4]

*.bioc.xml → the BioC-formatted CHEMDNER corpus - see chemdner.key and BioC.dtd


3) Plain-text abstracts
-----------------------

This file contains plain-text, UTF8-encoded PubMed abstracts in a 
tab-separated format with the following three columns:

1- Article identifier (PMID, PubMed identifier)
2- Title of the article
3- Abstract of the article

The test set abstracts consists in a set of 20,000 abstracts containing the 3000 
test set abstracts in addition to a background collection of 17,000 random
abstracts from PubMed. We added this background set to avoid any manual
correction of the predictions.


4) Plain-text annotations
-------------------------

These files contain manually generated annotations of chemical entities of the test dataset.

They contain tab-separated fields with:

1- Article identifier (PMID)
2- Type of text from which the annotation was derived (T: Title, A: Abstract)
3- Start offset
4- End offset
5- Text string of the entity mention
6- Type of chemical entity mention (ABBREVIATION,FAMILY,FORMULA,IDENTIFIERS,MULTIPLE,SYSTEMATIC,TRIVIAL)


5) chemdner_abs_test_pmid_label.txt
-----------------------------------

This file contains the classification of the provided evaluation abstracts ("test set")
into the 3,000 annotated abstracts and the background collection of 17,000 random
abstracts from PubMed. 
Column 1 corresponds to the PMID of the abstract, and column 2 to the abstract label. 
If the abstract is labelled 'Y' it was part of the manually revised test set, if the 
label is 'N' it is part of the PubMed background collection.


6) CDI test set data file
-------------------------

Test data Gold Standard file for the Chemical document indexing (CDI) sub-task: 

cdi_ann_test_13-09-13.txt

Given a set of documents, for this subtask, the participants were asked to return for each of them a ranked list 
of chemical entities described within each of these documents. 

It consists of tab-separated fields containing:

1- Article identifier (PMID)
2- Text string of the entity mention

An example is shown below:
21723361	docosahexaenoic acid
21723361	docosahexaenoic acids
21723361	methylmercury
21838705	MonoCarboxylate
21838705	darunavir
21838705	ritonavir
21838705	saquinavir


7) CEM test set data file
-------------------------

Test data Gold Standard file for the Chemical entity mention recognition (CEM) sub-task: 

cem_ann_test_13-09-13.txt


Given a set of documents, for this subtask, the participants had to return the start and end indices 
corresponding to all the chemical entities mentioned in this document. 

It consists of tab-separated columns containing:

1- Article identifier (PMID)
2- Offset string consisting in a triplet joined by the ':' character. You have to provide the text type (T: Title, A:Abstract), the start offset and the end offset.

An example illustrating the format is shown below:

21723361	A:100:113
21723361	A:46:67
21723361	T:11:31
21723361	T:36:49
21838705	A:1070:1085
21838705	A:1419:1429
21838705	A:1434:1443
21838705	A:1765:1774


8) Additional comments
----------------------

To evaluate the performance of your system we recommend you to use the
BioCreative evaluation library scripts. You can also directly download 
it from the BioCreative Resources page at:

http://www.biocreative.org/resources/biocreative-ii5/evaluation-library/

This webpage explains in detail how to install the library and how it works. 

For both of the tasks you should use the --INT evaluation option like shown below:

bc-evaluate --INT prediction_file evaluation_file

As the --INT option is chosen by default, you can also run this script without the
argument:

bc-evaluate prediction_file evaluation_file

Example evaluation files for both subtasks were described above.


A) Prediction format for the CDI subtask

Please make sure that your predictions are compliant with the formatting information provided for the --INT option of the evaluation library (The webpage and the bc-evaluate -h and bc-evaluate -d option provide you with more details.)

In short, you have to provide a tab-separated file with:

1- Article identifier
2- The chemical entity mention string
3- The rank of the chemical entity returned for this document
4- A confidence score

Example cases are provided online in the CHEMDNER sample set (June 25, 2013)

(http://www.biocreative.org/resources/corpora/bc-iv-chemdner-corpus/#bc-iv-chemdner-corpus:downloads)

An example prediction for the sample set is shown below:

6780324	LHRH	1	0.9
6780324	FSH	2	0.857142857143
6780324	3H2O	3	0.75
6780324	(Bu)2cAMP	4	0.75
6780324	vitro	5	0.666666666667
6780324	plasminogen	6	0.5
6780324	ethylamide	7	0.5
6780324	beta-3H]testosterone	8	0.5
6780324	NIH-FSH-S13	9	0.5
6780324	D-Ser-(But),6	10	0.5
6780324	4-h	11	0.5
6780324	3-isobutyl-l-methylxanthine	12	0.5
2231607	thymidylate	1	0.666666666667
2231607	acid	2	0.666666666667
2231607	TS	3	0.666666666667


B) Prediction format for the CEM subtask

Please make sure that your predictions are compliant with the formatting information provided for the --INT option of the evaluation library (The webpage and the bc-evaluate -h and bc-evaluate -d option provide you with more details.)

In short, you have to provide a tab-separated file with:

1- Article identifier (PMID)
2- Offset string consisting in a triplet joined by the ':' character. You have to provide the text type (T: Title, A:Abstract), the start offset and the end offset.
3- The rank of the chemical entity returned for this document
4- A confidence score

Example from the sample set (from June 25, 2013) is shown below:

6780324	A:104:107	1	0.5
6780324	A:1136:1147	2	0.5
6780324	A:1497:1500	3	0.5
6780324	A:162:167	4	0.5
6780324	A:17:21	5	0.5
6780324	A:319:330	6	0.5
6780324	A:448:452	7	0.5


9) CHEMDNER silver standard corpus
Corpus of a recent set of 17,000 randomly selected Pubmed abstracts (silver.abstracts.txt)
and automated team predictiosn for this collection (silver.predictions.txt)

10) CHEMDNER chemical disciplines subsets.
The file: 'chemdner_chemical_disciplines.txt' specifies subsets of abstracts that were selected based on
subject categories from the ISI Web of Knowledge relevant to various chemistry-related disciplines. The label
1 means that this PMID is associated to the corresponding chemical discipline column header. The following
disciplines were included:

BIOCHEMISTRY
CHEMISTRY_APPLIED
CHEMISTRY_MEDICINAL
CHEMISTRY_MULTIDISCIPLINARY
CHEMISTRY_ORGANIC
CHEMISTRY_PHYSICAL
ENDOCRINOLOGY
ENGINEERING_CHEMICAL
PHARMACOLOGY
POLYMER_SCIENCE
TOXICOLOGY

