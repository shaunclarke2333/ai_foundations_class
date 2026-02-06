import pandas as pd

file_path = "./documents_project4/computers.csv"

# Reading in CSV data as dataframe
df = pd.read_csv(file_path)
test = []
# Looping through the dataframe getting the row index and row data
for index, row in df.iterrows():
    # Empty list to hold the data for each row
    lines: list =[]
    # replacing all Na cells with Unknown
    row = row.fillna('Unknown')
    # looping through the data dictionary for each row
    for column, value in row.items():
        # adding the 
        lines.append(f"{column}: {value}")
    content = "\n".join(lines)

    test_dict = {
        "row": index,
        "content": content,
        "source": file_path,
        "page": None

    }

    test.append(test_dict)

print(test)
    
