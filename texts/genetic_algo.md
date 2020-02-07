# Genetic algorithms

For creating a genetic algorithm there're the following steps:  

- Creating an initial population
- Dealing with a fitness function
- Choosing the parents from whom the child will 'born'
- Crossing parents [crossover]
- Making mutations
  
## What is the fitness function

The fitness function determines how close the result is from what we want. For example, let's suppose we want to write the word `jabuticaba`. So considering the initial population the set of words above:

- `aksdmlkasmd` | Score: 0
- `jubasdjf`    | Score: 2
- `jobutdiagd`  | Score: 5
- `jaboticuba`  | Score: 8

Our fitness function sum the number of letters there are equal and in the same place of letters from jabuticaba.

## Selecting parents

Basically, we must choose the members of the population with the highest scores. In this case, it would be `jobutdiagd` and `jaboticuba`.

## Making the crossover 

So, our crossover can be all the combinations of the words `jobutdiagd` and `jaboticuba`. We can select, for example, 3 two letters from the firt word and substitute it in the second. The code in python would be something like:

```python
letter_1 = 'jobutdiagd'
letter_2 = 'jaboticuba'
for position, letter in enumerate(letter_1): 
    for position_, letter_ in enumerate(letter_1): 
        copy = letter_2
        copy = copy[:position] + letter + copy[position+1:]
        copy = copy[:position_] + letter_ + copy[position_+1:]
        print(copy)
```

In the end we gonna have a sequence like: 

- jaboticuba
- joboticuba
- ..
- jabutiiuba
- jabuticaba
- jabuticuga
- jabuticubd
- jaboticuba
- ...
- jaboticubd
- jabotdcubd
- jabotiiubd
- jaboticabd
- jaboticugd
- jaboticubd

It already generated the word `jabuticaba` for this example.

## Mutation

It might be the case in one generic algorithm that our generated population does not contribute for evolution. In this case, we can create a new function to create new 'random' variables from the population. It might be, for example, inverting two words and making the crossover between them.

## When the code stops? 

When we achieve the desired result.

## Fonts

[MediumArticle](https://medium.com/analytics-vidhya/understanding-genetic-algorithms-in-the-artificial-intelligence-spectrum-7021b7cc25e7)