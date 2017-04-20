## Synopsis

Hi, and thank you for visiting my little classification engine. It is coded with help of the Python lib NumPy, and has a total of five classifiers implemented.

## Motivation

This project was created with educational and learning purposes.

## How it works

This engine loads a dataset, parses it, and divides it in N partitions(depending on the Validation Type selected).
After this, a certain amount of the total dataset is trained with one of the following algorithms(implemented by me):
* Prior Algorithm.
* Naive-Bayes
* K-Neighbours
* Logistic Regression
* Genetic algorithm

The other part of the set is tested and, with the method "errores", we get the total Error Rate Obtained.

The format of the dataset can be checked in "example.data", located in this repository.
But basically, the format must be like this:

`<Number of tuples>`

`<NameOf1stAttribute,NameOf2ndAttribute...>`

`<Continuo/Nominal,Continuo/Nominal...>`

`<Data, with as many columns as number of attributes, PLUS the column of the real class>`

