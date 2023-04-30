from pylab import *
import matplotlib.pyplot as plt
import os
import pyAgrum as gum
import graphviz as gv
    
bn=gum.BayesNet('DropOutPredictor')
print(bn)
# Adding variables

school_performance=bn.add(gum.LabelizedVariable('School Performance','school_performance', ['low', 'medium', 'high']))
academic_performance=bn.add(gum.LabelizedVariable('Academic Performance','academic_performance', ['low', 'medium', 'high']))
attendance=bn.add(gum.LabelizedVariable('Attendance','attendance', ['low', 'medium', 'high']))
family_support=bn.add(gum.LabelizedVariable('Family Support', 'family_support', ['low', 'medium', 'high']))
home_environment=bn.add(gum.LabelizedVariable('Home environment', 'family_support', ['low', 'medium', 'high']))
parental_support=bn.add(gum.LabelizedVariable('Parental Support', 'parental_support', ['low', 'medium', 'high']))
financial_support=bn.add(gum.LabelizedVariable('Financial Support', 'financial_support', ['low', 'medium', 'high']))
government_support=bn.add(gum.LabelizedVariable('Government Support', 'government_support', ['low', 'high']))
social_support=bn.add(gum.LabelizedVariable('Social Support', 'social_support', ['low', 'medium', 'high']))
likelihood_of_dropout=bn.add(gum.LabelizedVariable('Likelihood of Dropout', 'likelihood_of_dropout', ['low', 'medium', 'high']))
age = bn.add(gum.LabelizedVariable('Age','age',['12','13','14','15','16','17']))
race = bn.add(gum.LabelizedVariable('Race','race',['black','white','asian','hespanic','others']))
Income = bn.add(gum.LabelizedVariable('Income','income',['low','medium','high']))
print (bn)

arcs = [(school_performance, likelihood_of_dropout),(academic_performance, school_performance), (attendance, school_performance), (family_support, likelihood_of_dropout),(home_environment,family_support),(age,home_environment),(race,home_environment), (parental_support, family_support), (financial_support, likelihood_of_dropout),(Income,financial_support), (government_support, financial_support),(social_support, likelihood_of_dropout)]
for arc in arcs:
    bn.addArc(*arc)
# Define the conditional probability tables
# I think we get this from data
    
    
# Compile the Bayesian network
ie = gum.LazyPropagation(bn)
    
# Set the evidence (observed nodes)
#ie.setEvidence({'Academic Performance': 'low'})
    
# Compute the posterior probability of the likelihood of dropout - inference
# ie.makeInference()
# posterior = ie.posterior(likelihood_of_dropout)
# print(posterior)

# Create a Graphviz source from the Bayesian network
dot = bn.toDot()
# Display the graph
graph = gv.Source(dot)
graph.render('dropouts')
