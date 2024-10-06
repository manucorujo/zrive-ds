```python
import pandas as pd

abandonded_carts = pd.read_parquet("/home/manucorujo/zrive-data/abandoned_carts.parquet")
inventory = pd.read_parquet("/home/manucorujo/zrive-data/inventory.parquet")
orders = pd.read_parquet("/home/manucorujo/zrive-data/orders.parquet")
regulars = pd.read_parquet("/home/manucorujo/zrive-data/regulars.parquet")
users = pd.read_parquet("/home/manucorujo/zrive-data/users.parquet")

```
