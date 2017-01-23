# EnronMachineLearning
The Enron scandal (https://en.wikipedia.org/wiki/Enron_scandal) was one of the more memorable financial scandals in recent American history. In the course of the government investigation of Enron, a large body of emails was made public. This corpus is a treasure trove of actual emails written between corporate executives, offering up the raw data to help hone machine learning techniques.

Approximately 150 people have their emails catalogued in this collection, but naturally, they were not all involved in committing fraud. In the course of the investigation, there were those who were indicted by the government, those who settled with the government without admitting guilt, and those who testified on behalf of the government in exchange for immunity. We will consider a person who fits into one of these three categories to be a “person of interest”. A USA Today article from 2005 handily helped to identify these persons of interest:
http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm

Using a combination of financial data and feature extraction from the Enron emails, I was able to use a supervised machine learning algorithm to create a classifier with 0.60 recall and 1.00 precision.
