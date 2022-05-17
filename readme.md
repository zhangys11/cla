# pyCLAMs

pyCLAMs: An integrated Python toolkit for classifiability analysis [J]. SoftwareX, Volume 18, June 2022, 101007, doi: 10.1016/j.softx.2022.101007  
https://doi.org/10.1016/j.softx.2022.101007

# Installation 

pip install pyCLAMs
pip install rpy2
You should also have the R runtime with the ECol library (https://github.com/lpfgarcia/ECoL) installed.

# How to use 

Download the sample dataset from the /data folder
Use the following sample code to use the package:

<pre>
  # import the library
  from pyCLAMs import clams

  # load the dataset or generate a toy dataset by X,y = mvg(md = 2)
  df = pd.read_csv('sample.csv')
  X = np.array(df.iloc[:,:-1]) # skip first and last cols
  y = np.array(df.iloc[:,-1])

  # get all metrics
  clams.get_metrics(X,y) # Return a dictionary of all metrics

  # get metrics as JSON
  clams.get_json(X,y)

  # get an html report and display in Jupyter notebook
  from IPython.display import display, HTML
  display(HTML(clams.get_html(X,y)))
</pre>

# Extra Material
A more friendly GUI tool based on pyCLAMs can be accessed at http://spacs.brahma.pub/research/CLA

# Metrics added since the original publication

classification.Mean_KLD - mean KLD (Kullback-Leibler divergence) between ground truth and predicted one-hot encodings  
correlation.r2 - R2, the R-squared effect size
test.CHISQ, test.CHISQ.log10, test.CHISQ.CHI2 - Chi-squared test
