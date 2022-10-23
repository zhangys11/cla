# pyCLAMs

pyCLAMs: An integrated Python toolkit for classifiability analysis [J]. SoftwareX, Volume 18, June 2022, 101007, doi: 10.1016/j.softx.2022.101007  
https://doi.org/10.1016/j.softx.2022.101007

# Warning

Since 0.3.x, we have reorganized the package structure. Any upper app should be revised accordingly.

# Installation 

pip install pyCLAMs
pip install rpy2
Install the R runtime and the ECol library (https://github.com/lpfgarcia/ECoL).  

  Run 'install.packages("ECoL")' in R. It will take very long time. You must wait for the installation to complete.     
  Sometimes, you may want to change the CRAN mirror. Under the "Packages" menu, click "Set CRAN Mirror".    
  After installation, you can check by R command 'installed.packages()'. 

# How to use 

Download the sample dataset from the /data folder
Use the following sample code to use the package:

<pre>
  # import the library
  import clams

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
  classification.McNemar, classification.McNemar.CHI2 - McNemar test on the groud-truth and classifier's prediction     


