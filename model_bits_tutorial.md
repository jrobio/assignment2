# Model Bits Tutorial
How to use the `deque()` and `dict()` collections as well as the `str.join()` function to create iteratively built bit
words to represent and conveniently describe your models.

## The Problem
One problem I came across while developing my models was keeping track of and organizing them. I found that I wanted
to build my models iteratively and to add complexity as I went through the modeling process. For example, I may want to 
target my data for logistic and linear analysis, in addition to running it through several types scaling, different
approaches to automatic feature selection, as well as regressors and classsifiers. 

As you can imagine, this problem grew exponentially with each step, and I found it difficult to keep it all in my head. 

## The Solution
The solution I settled on was to represent each model with a string of bits that is built over the life-cycle of model 
development. This not only provided a clear mental model for approaching model development, but it also provided each 
model with an easily referenced string of bits that wholly defined how it was developed.

### An Example
```
0th bit [-1]:           0 - logistic regression
                        1 - linear regression
1st bit [-2]:           0 - grad_year int
                        1 - grad_year binned
2nd bit [-3]:           0 - cum_donation float
                        1 - cum_donation binned
3rd-5th bit [-6:-3]:    000 - no automatic feature selection
                        001 - chi square filtering (chi)
                        010 - Random Forest Importance (rfi)
                        011 - Recursive Feature Elimination Cross Validation (rfe)
                        100 - Forward Feature Elimination (ffe)
6th-7th bit [-8:-6]:    00 - unscaled
                        01 - MinMaxScaler
                        10 - StandardScaler
                        11 - RobustScaler
8th bit [-9]:           0 - Cross Fold Validation
                        1 - Stacking
                
                        The Model at a Glance:
010011110               A logistic target, with both binned features, filtered by recurcive feature elimination with 
                        cross validation, using the Standard Scaler, and Cross Folded.
                        **read from right to left 
```

The tools I used to develop this approach are the `deque()` and `dict()` collections as well as the `str.join()` 
function.

## The Code
This block iterates through `dict_3bit` containing all models developed thus far (of up to 2<sup>3</sup> complexity) 
creates 5 versions of each and appends them to `dict_6bit`.

```python
# import deque because you'll need it
import pandas as pd
from collections import deque

# A collection containing models of 2^3 complexity
dict_3bit = {
    '110': {
        'X': pd.DataFrame(),
        'y': pd.DataFrame()
    }
}

# 3rd-5th bit: Automatic Feature Selection (afs)
# 000 - no automatic feature selection
# 001 - chi square filtering (chi)
# 010 - Random Forest Importance (rfi)
# 011 - Recursive Feature Elimination Cross Validation (rfe)
# 100 - Forward Feature Elimination (ffe)

def get_chi(bit_str: str, d):
    pass


def get_rfi(bit_str: str, d):
    pass


def get_rfecv(bit_str, d):
    pass


def get_ffe(bit_str: str, d):
    pass


dict_6bit = dict()  # dict that will contain models of up to 2^6 complexity

for k_model, v_data in dict_3bit.items():
    for auto in [('000', None), ('001', get_chi), ('010', get_rfi),
                 ('011', get_rfecv), ('100', get_ffe)]:
        model_bit = deque(k_model)
        data = v_data.copy()

        # skips no automatic feature selection
        if auto[0] == '000':
            model_bit.appendleft(auto[0])
        # skips chi-squaring linear regressions
        elif (auto[1] is get_chi) and (bit_str[-1] == '1'):
            model_bit.appendleft('000')
        else:
            model_bit.appendleft(auto[0])
            data = auto[1](model_bit, data)
        bit_str = ''.join(model_bit)
        dict_6bit.update({bit_str: data})
```
### Queue? Stack? Deque? What do they dueue?
#### queue()
A `queue` is a data collection that enforces a FIFO (first in, first out) policy on its elements, which means you can 
`.append()` elements to the end, and `.pop()` elements from the front. 

#### stack()
A queue is the opposite of a `stack`, which has a LIFO (last in, first out), which means the first element becomes the 
bottom of a `stack` which subsequent elements are pushed on top of. 

#### deque()
A `deque` is a double-ended `queue` and is a generalization of both the `queue` and `stack`, which means it can be made
to operate like either. Crucially, the `deque` implements an `.appendleft()` and `.appendright()` function.

##### Working with Deque
```python
# Deque needs to be imported from the collections library
from collections import deque

# let's create a string
# Fun Fact: strings in python aren't mutable!
test = 'Robert'
print(test)

>>> Robert

# let's create a deque and see what it does when we feed it a string
de = deque(test)
print(de)

# Whoa! So hip and cool
>>> deque(['R', 'o', 'b', 'e', 'r', 't'])

# let's add something to the left
de.appendleft('neato')
print(de)

# Weeeeeirrrdd! But still hip and cool
>>> deque(['neato', 'R', 'o', 'b', 'e', 'r', 't'])

# let's collapse it back into a string
test = ''.join(de)
print(test)

>>> neatoRobert

# and back to a deque
print(deque(test))

>>> deque(['n', 'e', 'a', 't', 'o', 'R', 'o', 'b', 'e', 'r', 't'])
```
With the above functionality, it is possible to implement the approach I took in my assignment, so let's get to that:

## The Tutorial

1. Iterate through `dict_3bit`:
    ```python
    dict_3bit = {
        '110': {
            'X': pd.DataFrame(),
            'y': pd.DataFrame()
        }
    }
    
    for k_model, v_data in dict_3bit.items():
    ```
2. For each type of feature selection (`None, chi, rfi, rfecv, ffe`):
    ```python
        for auto in [('000', None), ('001', get_chi), ('010', get_rfi),
                     ('011', get_rfecv), ('100', get_ffe)]:
    ```
3. Ingest the key (`k_model`) and turn it into a `deque` bit word representing the model. Assign the key's value too:
    ```python
            model_bit = deque(k_model)
            data = v_data.copy()
    ```
4. Skip models that won't receive any form of automatic feature selection. Encode this decision into the model's bit 
word:
    ```python
            if auto[0] == '000':
                model_bit.appendleft(auto[0])
    ```
5. Block Linear models from receiving chi-square filtering. Encode this decision into the model's bit word:
    ```python
            elif (auto[1] is get_chi) and (bit_str[-1] == '1'):
                model_bit.appendleft('000')
    ```
6. Send model's for automatic feature filtering. Encode this decision into the model's bit word:
    ```python
            else:
                model_bit.appendleft(auto[0])
                data = auto[1](model_bit, data)
    ```
7. Collapse the bit word into a string and assign higher complexity model to `dict_6bit`:
    ```python
            bit_str = ''.join(model_bit)
            dict_6bit.update({bit_str: data})
    ```
The resulting data-structure will look something like this:
```python
# a logistic model with both binned features that has received Random Forest Importance filtering
dict_6bit = {
    '010110': {
        'X': pd.DataFrame(),
        'y': pd.DataFrame(),
        'afs': pd.DataFrame()
    }
}
```
Where `'afs'` contains a DataFrame of features and their respective filtering scores. `X` is now a subset of significant
features. You're welcome to check out the 
<a href='https://github.com/jrobio/assignment2/blob/master/notebooks/assignment2.ipynb'>code</a> of my full assignment 
if you are interested in how I implemented my filtering methods.