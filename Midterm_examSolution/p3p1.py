import pandas as pd
import cudf 


# Create the DataFrame
data = {
    'kind': ['cat', 'dog', 'cat', 'dog'],
    'height': [9.1, 6.0, 9.5, 34.0],
    'weight': [7.9, 7.5, 9.9, 198.0]
}
df = pd.DataFrame(data)

# Group by 'kind' and find min and max height
result = df.groupby('kind')['height'].agg(['min', 'max'])

# Print the results
print("--------Using Pandas:-----------")
print(result)
print('\n\n')

## Create the cuDF DataFrame
data_gpu = {
    'kind': ['cat', 'dog', 'cat', 'dog'],
    'height': [9.1, 6.0, 9.5, 34.0],
    'weight': [7.9, 7.5, 9.9, 198.0]
}
df_gpu = cudf.DataFrame(data_gpu)

# Group by 'kind' and find min and max height
result_gpu = df_gpu.groupby('kind')['height'].agg(['min', 'max'])

# Print the results
print("\n --------------Using cuDF:---------------")
print(result_gpu)
print('\n\n')
