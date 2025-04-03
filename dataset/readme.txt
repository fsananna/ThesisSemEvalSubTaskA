subtask_a_train.tsv
	Tab-separated dataset.

	Columns:
	**compound**		The potentially idiomatic noun compound to which the other data relates. This compound will appear (once) within **sentence**.
	**subset**			Indicates the data subset to which the item belongs. Values: {Train, Sample, Dev, Test}. Only {Train, Sample} will appear in this dataset; the latter indicates that the item was included in the smaller sample data provided on the website. All items in this dataset can be used for training purposes.    
	**sentence_type**	Indicates which sense of **compound** is used in **sentence**. Values: {idiomatic, literal}. This field will _not_ be included in dev and test data and should not be consumed by participating systems. It is provided here for information and analysis purposes, and as a possible target for system component training.
	**sentence**		Target sentence in which **compound** appears.
	**expected_order**	List of image names. This is the target output for the shared task which will be used for evaluation. This field will not be included in dev and test datasets.
	**image{n}_name**	Filename of the nth candidate image. This file is located in the subfolder which shares its name with **compound**, e.g. "green fingers/10027562830.png".
	**image{n}_caption**	Machine-generated descriptive caption of the nth candidate image. This is intended for participants who do not wish to perform image processing, but may be used to supplement the image files if desired. Note that the descriptions may not accurately reflect the intended content of the image, as they are the output of automatic captioning.

train: 
	train dataset
eval: 
	development dataset,
	Note that the fields sentence_type and expected_order are blank in the development dataset,
test: 
	test dataset
xeval: 
	extended evaluation dataset 


Note: 
	{n} for image name and caption ranges from 1 to 5 inclusive. Image name and caption fields are ordered by the (randomised) image filename.


Subfolders:
	One subfolder for each target **compound**. Each subfolder contains 5 image files, corresponding to the entries in **image1_name** to **image5_name**.
	
Note that: 
	the sentence_type and expected_order fields should not be consumed by systems and are made available for analysis purposes.