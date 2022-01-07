---- pyCLAMs ----

pyCLAMs: an integrated Python toolkit for classifiability analysis
A more friendly GUI tool using pyCLAMs can be accessed at http://spacs.brahma.pub/research/CLA

---- Installation ----

pip install pyCLAMs
pip install rpy2
You should also have the R runtime with the ECol library (https://github.com/lpfgarcia/ECoL) installed.

---- How to use ----

Download the sample dataset from the /data folder
Use the following sample code to use the package:

# import the library
from pyCLAMs.pyCLAMs import *

# load the dataset or generate a toy dataset by X,y = mvg(md = 2)
df = pd.read_csv('sample.csv')
X = np.array(df.iloc[:,:-1]) # skip first and last cols
y = np.array(df.iloc[:,-1])

# get all metrics
get_metrics(X,y) # Return a dictionary of all metrics

# get metrics as JSON
get_json(X,y)

# get an html report and display in Jupyter notebook
from IPython.display import display, HTML
display(HTML(get_html(X,y)))