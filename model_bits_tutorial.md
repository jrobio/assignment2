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
0th bit:        0 - logistic regression
                1 - linear regression
1st bit:        0 - a float feature as float
                1 - a float feature as binned and dummied
2nd bit:        0 - an int feature as int
                1 - an int feature as binned and dummied
3rd bit:        0 - features not filtered by chi
                1 - features filtered by chi > 3.8
4th-5th bit:    00 - no scaling
                01 - MinMaxScaler
                10 - StandardScaler
                11 - RobustScaler
6th-7th bit:    00 - Random Tree Importance
                01 - Recurcive Feature Elimination
                10 - Forward Feature Elimination
8th bit         0 - Cross Fold Validation
                1 - Stacking
                

010011110       A logistic target, with both binned features, filtered with chi > 3.8, scaled with the MinMaxScaler,
                with Forward Feature Elimation, and with Cross Fold Validation.
                **read from right to left 
```

The tools I used to develop this approach are the `deque()` and `dict()` collections as well as the `str.join()` 
function.

## The Code
```python
# import deque because you'll need it
from collections import deque

# 4th-5th bit: Scaling
# 00 - unscaled
# 01 - MinMaxScaler
# 10 - StandardScaler
# 11 - RobustScaler

dict_6bit = defaultdict()

for k_model, v_data in dict_4bit.items():
    for scale in [('00',None),('01',MinMaxScaler()),('10',StandardScaler()),
                  ('11',RobustScaler())]:
        model_bit = deque(k_model)
        leaves = v_data

        if model_bit[-1] == '0':
            reg = 'log'
        else:
            reg = 'lin'

        leaves.update({'y_scaled': Xy_dict[reg]['y'],
                       'X_scaled': Xy_dict[reg]['X']})

        if scale[0] == '00':
            model_bit.appendleft(scale[0])

        else:
            if reg == 'lin':
                y_scaled = scale[1].fit_transform(np.array(Xy_dict[reg]['y'])
                                                  .reshape(-1,1))
                leaves.update({'y_scaled':y_scaled})

            model_bit.appendleft(scale[0])
            X_scaled = scale[1].fit_transform(
                Xy_dict[reg]['X'][leaves['features']])
            leaves.update({'X_scaled': X_scaled})
        bit_str = ''.join(model_bit)
        dict_6bit.update({bit_str: leaves})

```