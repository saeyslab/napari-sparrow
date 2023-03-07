 # Usecases
 
 The goal of this document is to give an overview on where to start when starting with SPArrOW.
 I will try to update this document regularly every time I fix something. 
 
 ## Installation
 
Please follow this installation in the readme.md. Currently installation is only possible by cloning the repo. Make sure you start from the lottep branch, as this is the branch that is most up-to-date.
At some point we will update main, but currently we don't want to break code of people still working on main. 

## Where to start

A good first practice is to run the pipeline once on data where it was created for. 
We advise to start with the RESOLVE data on mouse liver. 
The notebook for this is napari-sparrow/experiments/script_liver_sparrow_generalized.ipynb. (only on lottep!)
The data itself can be downloaded from this site: 
https://cloud.irc.ugent.be/public/index.php/s/HrXG9WKqjqHBEzS.
The dataset used in the examples is mouse liver A1-1. You need the DAPI-stained image and the txt file. 

A second option would then be to have a look at the vizgen notebook, especially when interested in analysing vizgen data. 
In this notebook, the segmentation is shown on polyT instead of on DAPI. The data can be downloaded from the links provided. 
PLease start first with the subset, before moving on to the complete dataset. 
napari-sparrow/experiments/VizgenpolyT_Tutorial.ipynb only on lottep branch! 


## Known issues

Currently, the expansion of the cells is broken, and I am working at it to fix it. It should be fixed by 9th of March for sure.
Secondly, the joint segmentation of DAPI-stained image and the polyT-staining is broken too. This hopefully will be fixed soon too. 

## Having extra issues
Please just create a github issue. I know the list is long at the moment,but we possibly can give priority to things you might need.
Also, there is a chance we have done something similar already and can provide you with some help. 




 
 
