# Milestone 1: Exploration Phase


```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

```


```python
CSV_PATH = "/home/manucorujo/zrive-data/feature_frame.csv"
```


```python
df = pd.read_csv(CSV_PATH)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB



```python
information_cols = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
target_col = 'outcome'
feature_cols = [col for col in df.columns if col not in information_cols + [target_col]]

categorical_cols = ['product_type', 'vendor']
binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
numerical_cols = [col for col in feature_cols if col not in categorical_cols + binary_cols]
```

## 1st task: filter the data to only those orders with 5 items or more to build a dataset to work with.


```python
# Sum outcome (A product is in the order if its outcome is 1)
count_products = df.groupby("order_id").outcome.sum().reset_index()
count_products.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2807985930372</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2808027644036</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2808099078276</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2808393957508</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2808429314180</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
filtered_orders = count_products[count_products.outcome >= 5]
```


```python
filtered_df = df[df['order_id'].isin(filtered_orders['order_id'])]
filtered_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2163953 entries, 0 to 2880547
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 462.3+ MB


## 2nd task: Split data into training, validation and test


```python
filtered_df.shape
```




    (2163953, 27)




```python
filtered_df.groupby('order_id').outcome.sum().reset_index().shape
```




    (2603, 2)



There are 2163953 products spread over 2603 orders. In my opinion, a good approach could be a temporary-based split (in order to prevent from infomation leakage), using a typical 70/20/10 distribution.


```python
# Get how many orders are performed each day
daily_orders = filtered_df.groupby('order_date').order_id.nunique()
daily_orders.head()
```




    order_date
    2020-10-05 00:00:00     3
    2020-10-06 00:00:00     7
    2020-10-07 00:00:00     6
    2020-10-08 00:00:00    12
    2020-10-09 00:00:00     4
    Name: order_id, dtype: int64




```python
total_orders = daily_orders.sum()
percentage_orders = pd.DataFrame(columns=['date', 'percentage'])
rows = []
number_orders = 0
for _, row in daily_orders.reset_index().iterrows():
    date = row.iloc[0]
    number_orders += row.iloc[1]
    rows.append(
        {'date': date, 'percentage': number_orders / total_orders}
    )
percentage_orders = pd.DataFrame(rows)

print(percentage_orders.head())

train_val_cut = percentage_orders[percentage_orders.percentage <= 0.7].iloc[-1]
val_test_cut = percentage_orders[percentage_orders.percentage <= 0.9].iloc[-1]

print(f"Train set from: {daily_orders.index.min()}")
print(f"Train set to: {train_val_cut.date}")
print(f"Validation set from: {train_val_cut.date}")
print(f"Validation set to: {val_test_cut.date}")
print(f"Test set from: {val_test_cut.date}")
print(f"Test set to: {daily_orders.index.max()}")

```

                      date  percentage
    0  2020-10-05 00:00:00    0.001153
    1  2020-10-06 00:00:00    0.003842
    2  2020-10-07 00:00:00    0.006147
    3  2020-10-08 00:00:00    0.010757
    4  2020-10-09 00:00:00    0.012294
    Train set from: 2020-10-05 00:00:00
    Train set to: 2021-02-04 00:00:00
    Validation set from: 2021-02-04 00:00:00
    Validation set to: 2021-02-22 00:00:00
    Test set from: 2021-02-22 00:00:00
    Test set to: 2021-03-03 00:00:00


With the dates to split the data selected, is time to create the actual datasets


```python
train_set = filtered_df[filtered_df.order_date <= train_val_cut.date]
val_set = filtered_df[(filtered_df.order_date > train_val_cut.date) & (filtered_df.order_date <= val_test_cut.date)]
test_set = filtered_df[filtered_df.order_date > val_test_cut.date]
```

Divide dataset into features and target


```python
x_train = train_set[feature_cols]
y_train = train_set[target_col]

x_val = val_set[feature_cols]
y_val = val_set[target_col]

x_test = test_set[feature_cols]
y_test = test_set[target_col]
```

## 3rd task: Train models

I will start using a model with just numeric and binary variables, as they don't need preprocessing. Then, I can compare it with models that include all the variables. Due to the problem is a binary classification I will use Logistic Regression (lineal model)


```python
train_variables = numerical_cols + binary_cols

# Use only numerical and binary columns to train the model (Ridge regression)
x_train = x_train[train_variables]
x_val = x_val[train_variables]
x_test = x_test[train_variables]
model = LogisticRegression()
model.fit(x_train, y_train)
# CURRENT STATUS: Logistic regression not converging. Still a lot of work to do.
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[57], line 8
          6 x_test = x_test[train_variables]
          7 model = LogisticRegression(max_iter=1000)
    ----> 8 model.fit(x_train, y_train)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/sklearn/base.py:1473, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1466     estimator._validate_params()
       1468 with config_context(
       1469     skip_parameter_validation=(
       1470         prefer_skip_nested_validation or global_skip_validation
       1471     )
       1472 ):
    -> 1473     return fit_method(estimator, *args, **kwargs)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:1350, in LogisticRegression.fit(self, X, y, sample_weight)
       1347 else:
       1348     n_threads = 1
    -> 1350 fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer=prefer)(
       1351     path_func(
       1352         X,
       1353         y,
       1354         pos_class=class_,
       1355         Cs=[C_],
       1356         l1_ratio=self.l1_ratio,
       1357         fit_intercept=self.fit_intercept,
       1358         tol=self.tol,
       1359         verbose=self.verbose,
       1360         solver=solver,
       1361         multi_class=multi_class,
       1362         max_iter=self.max_iter,
       1363         class_weight=self.class_weight,
       1364         check_input=False,
       1365         random_state=self.random_state,
       1366         coef=warm_start_coef_,
       1367         penalty=penalty,
       1368         max_squared_sum=max_squared_sum,
       1369         sample_weight=sample_weight,
       1370         n_threads=n_threads,
       1371     )
       1372     for class_, warm_start_coef_ in zip(classes_, warm_start_coef)
       1373 )
       1375 fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
       1376 self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/sklearn/utils/parallel.py:74, in Parallel.__call__(self, iterable)
         69 config = get_config()
         70 iterable_with_config = (
         71     (_with_config(delayed_func, config), args, kwargs)
         72     for delayed_func, args, kwargs in iterable
         73 )
    ---> 74 return super().__call__(iterable_with_config)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/joblib/parallel.py:1918, in Parallel.__call__(self, iterable)
       1916     output = self._get_sequential_output(iterable)
       1917     next(output)
    -> 1918     return output if self.return_generator else list(output)
       1920 # Let's create an ID that uniquely identifies the current call. If the
       1921 # call is interrupted early and that the same instance is immediately
       1922 # re-used, this id will be used to prevent workers that were
       1923 # concurrently finalizing a task from the previous call to run the
       1924 # callback.
       1925 with self._lock:


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/joblib/parallel.py:1847, in Parallel._get_sequential_output(self, iterable)
       1845 self.n_dispatched_batches += 1
       1846 self.n_dispatched_tasks += 1
    -> 1847 res = func(*args, **kwargs)
       1848 self.n_completed_tasks += 1
       1849 self.print_progress()


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/sklearn/utils/parallel.py:136, in _FuncWrapper.__call__(self, *args, **kwargs)
        134     config = {}
        135 with config_context(**config):
    --> 136     return self.function(*args, **kwargs)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:455, in _logistic_regression_path(X, y, pos_class, Cs, fit_intercept, max_iter, tol, verbose, solver, coef, class_weight, dual, penalty, intercept_scaling, multi_class, random_state, check_input, max_squared_sum, sample_weight, l1_ratio, n_threads)
        451 l2_reg_strength = 1.0 / (C * sw_sum)
        452 iprint = [-1, 50, 1, 100, 101][
        453     np.searchsorted(np.array([0, 1, 2, 3]), verbose)
        454 ]
    --> 455 opt_res = optimize.minimize(
        456     func,
        457     w0,
        458     method="L-BFGS-B",
        459     jac=True,
        460     args=(X, target, sample_weight, l2_reg_strength, n_threads),
        461     options={
        462         "maxiter": max_iter,
        463         "maxls": 50,  # default is 20
        464         "iprint": iprint,
        465         "gtol": tol,
        466         "ftol": 64 * np.finfo(float).eps,
        467     },
        468 )
        469 n_iter_i = _check_optimize_result(
        470     solver,
        471     opt_res,
        472     max_iter,
        473     extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
        474 )
        475 w0, loss = opt_res.x, opt_res.fun


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/scipy/optimize/_minimize.py:731, in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
        728     res = _minimize_newtoncg(fun, x0, args, jac, hess, hessp, callback,
        729                              **options)
        730 elif meth == 'l-bfgs-b':
    --> 731     res = _minimize_lbfgsb(fun, x0, args, jac, bounds,
        732                            callback=callback, **options)
        733 elif meth == 'tnc':
        734     res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,
        735                         **options)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/scipy/optimize/_lbfgsb_py.py:407, in _minimize_lbfgsb(fun, x0, args, jac, bounds, disp, maxcor, ftol, gtol, eps, maxfun, maxiter, iprint, callback, maxls, finite_diff_rel_step, **unknown_options)
        401 task_str = task.tobytes()
        402 if task_str.startswith(b'FG'):
        403     # The minimization routine wants f and g at the current x.
        404     # Note that interruptions due to maxfun are postponed
        405     # until the completion of the current minimization iteration.
        406     # Overwrite f and g:
    --> 407     f, g = func_and_grad(x)
        408 elif task_str.startswith(b'NEW_X'):
        409     # new iteration
        410     n_iterations += 1


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/scipy/optimize/_differentiable_functions.py:343, in ScalarFunction.fun_and_grad(self, x)
        341 if not np.array_equal(x, self.x):
        342     self._update_x(x)
    --> 343 self._update_fun()
        344 self._update_grad()
        345 return self.f, self.g


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/scipy/optimize/_differentiable_functions.py:294, in ScalarFunction._update_fun(self)
        292 def _update_fun(self):
        293     if not self.f_updated:
    --> 294         fx = self._wrapped_fun(self.x)
        295         if fx < self._lowest_f:
        296             self._lowest_x = self.x


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/scipy/optimize/_differentiable_functions.py:20, in _wrapper_fun.<locals>.wrapped(x)
         16 ncalls[0] += 1
         17 # Send a copy because the user may overwrite it.
         18 # Overwriting results in undefined behaviour because
         19 # fun(self.x) will change self.x, with the two no longer linked.
    ---> 20 fx = fun(np.copy(x), *args)
         21 # Make sure the function returns a true scalar
         22 if not np.isscalar(fx):


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/scipy/optimize/_optimize.py:79, in MemoizeJac.__call__(self, x, *args)
         77 def __call__(self, x, *args):
         78     """ returns the function value """
    ---> 79     self._compute_if_needed(x, *args)
         80     return self._value


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/scipy/optimize/_optimize.py:73, in MemoizeJac._compute_if_needed(self, x, *args)
         71 if not np.all(x == self.x) or self._value is None or self.jac is None:
         72     self.x = np.asarray(x).copy()
    ---> 73     fg = self.fun(x, *args)
         74     self.jac = fg[1]
         75     self._value = fg[0]


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/sklearn/linear_model/_linear_loss.py:281, in LinearModelLoss.loss_gradient(self, coef, X, y, sample_weight, l2_reg_strength, n_threads, raw_prediction)
        278 else:
        279     weights, intercept = self.weight_intercept(coef)
    --> 281 loss, grad_pointwise = self.base_loss.loss_gradient(
        282     y_true=y,
        283     raw_prediction=raw_prediction,
        284     sample_weight=sample_weight,
        285     n_threads=n_threads,
        286 )
        287 sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
        288 loss = loss.sum() / sw_sum


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-ojkehGB--py3.11/lib/python3.11/site-packages/sklearn/_loss/loss.py:255, in BaseLoss.loss_gradient(self, y_true, raw_prediction, sample_weight, loss_out, gradient_out, n_threads)
        252 if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
        253     gradient_out = gradient_out.squeeze(1)
    --> 255 self.closs.loss_gradient(
        256     y_true=y_true,
        257     raw_prediction=raw_prediction,
        258     sample_weight=sample_weight,
        259     loss_out=loss_out,
        260     gradient_out=gradient_out,
        261     n_threads=n_threads,
        262 )
        263 return loss_out, gradient_out


    KeyboardInterrupt: 



```python

```
