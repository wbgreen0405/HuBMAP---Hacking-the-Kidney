# <a id='3'>Introduction📔</a>
[Table of contents](#0.1)

Welcome to this new Kaggle competition. The [Human BioMolecular Atlas Program (HuBMAP)](https://hubmapconsortium.org/) is sponsored by The [National Institutes of Health (NIH)](https://www.nih.gov/). The primary task of HuBMAP is to catalyze the development of a framework for mapping the human body at a level of **glomeruli functional tissue units** for the first time in history. Hoping to become one of the world’s largest collaborative biological projects, HuBMAP aims to be an open map of the human body at the cellular level. **This competition, “Hacking the Kidney," starts by mapping the human kidney at single cell resolution.**

"**Your challenge is to detect functional tissue units (FTUs) across different tissue preparation pipelines.**"

Successful submissions will construct the tools, resources, and cell atlases needed to determine how the relationships between cells can affect the health of an individual.

## What is HuBMAP?

The focus of HuBMAP is understanding the intrinsic intra-, inter-, and extra- cellular biomolecular distribution in human tissue. HuBMAP will focus on fresh, fixed, or frozen healthy human tissue using in situ and dissociative techniques that have high-spatial resolution.

The Human BioMolecular Atlas Program is a consortium composed of diverse research teams funded by the [Common Fund at the National Institutes of Health](https://commonfund.nih.gov/HuBMAP) . HuBMAP values secure, open sharing, and collaboration with other consortia and the wider research community.

HuBMAP is developing the tools to create an open, global atlas of the human body at the cellular level. These tools and maps will be openly available, to accelerate understanding of the relationships between cell and tissue organization and function and human health.

## What is FTU?

An FTU is defined as a “three-dimensional block of cells centered around a capillary, such that each cell in this block is within diffusion distance from any other cell in the same block” (de Bono, 2013). 

The glomerulus (plural glomeruli) is a network of small blood vessels (capillaries) known as a tuft, located at the beginning of a nephron in the kidney. The tuft is structurally supported by the mesangium (the space between the blood vessels), composed of intraglomerular mesangial cells. The blood is filtered across the capillary walls of this tuft through the glomerular filtration barrier, which yields its filtrate of water and soluble substances to a cup-like sac known as Bowman's capsule. 

<br>

<div style="clear:both;display:table">
<img src="https://ohiostate.pressbooks.pub/app/uploads/sites/36/h5p/content/37/images/file-599206597bdbc.jpg" style="width:45%;float:left"/>
<img src="https://cdn.kastatic.org/ka-perseus-images/0e7bfc98302c3e45dc7ec73ab142566a57513ec3.svg" style="width:45%;float:left"/>
</div>

<br>

## Competition Goal

* The goal of this competition is the implementation of a successful and robust glomeruli FTU detector. Develop segmentation algorithms that identify **"Glomerulus"** in the PAS stained microscopy data. Detect functional tissue units (FTUs) across different tissue preparation pipelines.

* For each image we are given annotations in separate JSON file and also the annotations are RLE encoded in train.csv.

* We are segmenting **glomeruli FTU** in each image.

* Since this is segmentation task our evaluation metric is Dice Coefficient. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth.

## About Competition Data

The data is huge **(24.5 GB)**. The HuBMAP data used in this hackathon includes 11 fresh frozen and 9 Formalin Fixed Paraffin Embedded (FFPE) PAS kidney images. Stained microscopy employs histological stains such as H&E or PAS to improve resolution and contrast for visualization of anatomical structures such as tubules or glomeruli. Glomeruli FTU annotations exist for all 20 tissue samples. Some of these will be shared for training, and others will be used to judge submissions.

* The dataset is comprised of very large (>500MB - 5GB) TIFF files. 
* **"The training set"** has 8, and the public test set has 5 tiff files respectively. 
* **"The private test set"** is larger than the public test set.
* The training set includes annotations in both RLE-encoded and unencoded (JSON) forms. The annotations denote segmentations of glomeruli.

* **Both the training and public test sets also include anatomical structure segmentations. They are intended to help you identify the various parts of the tissue.**

We are provided with following files:

* For each of the 11 training images we have been provided with a JSON file. Each JSON file has:
   * A type (Feature) and object type id (PathAnnotationObject). Note that these fields are the same between all files and do not offer signal.
   * A geometry containing a Polygon with coordinates for the feature's enclosing volume
   * Additional properties, including the name and color of the feature in the image.
   * The IsLocked field is the same across file types (locked for glomerulus, unlocked for anatomical structure) and is not signal-bearing.

* train.csv contains the unique IDs for each image, as well as an RLE-encoded representation of the mask for the objects in the image. See the evaluation tab for details of the RLE encoding scheme. Note that we are also given annotations in JSON file for each image.

* HuBMAP-20-dataset_information.csv contains additional information (including anonymized patient data) about each image.

## What is RLE?

Run-length encoding (RLE) is a form of lossless data compression in which runs of data (sequences in which the same data value occurs in many consecutive data elements) are stored as a single data value and count, rather than as the original run.

## What we are prediciting?

Participants will develop segmentation algorithm that identify **"glomeruli "** in the PAS stained microscopy data. Detect functional tissue units (FTUs) across different tissue preparation pipelines. Participants are welcome to use other external data and/or pre-trained machine learning models in support of FTU segmentation. 

**We need to segment glomeruli in very large resolution Kidney images and annotations which are availabel as RLE encoded and as well as a JSON format.**

## Evaluation Metric: Dice Coefficient

Dice Coefficient is common in case our task involve **segmentation**. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. the Dice similarity coefficient for two sets X and Y is defined as:

$$\text{DC}(X, Y) = \frac{2 \times |X \cap Y|}{|X| + |Y|}.$$

where X is the predicted set of pixels and Y is the ground truth.
