import pandas as pd
import numpy as np
import itertools
import dask.dataframe as dd
from dask import delayed

def explode(df, columns_to_explode, debug=False):
    # Convert Pandas DataFrame to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=2)  # Adjust the number of partitions as needed

    # Define a function to explode a column
    @delayed
    def explode_column(column):
        return column.explode()

    # Explode each column in parallel
    exploded_columns = [explode_column(ddf[col]) for col in columns_to_explode]

    # Compute the results
    try:
        result_columns = dd.compute(*exploded_columns)
    # If TypeError: 'dict_values' object does not support indexing, try:
    except TypeError:
        result_columns = dd.compute(*list(exploded_columns))
    # Combine the results with regular columns
    result_df = pd.concat([df.drop(columns=columns_to_explode)] + list(result_columns), axis=1)

    return result_df
    