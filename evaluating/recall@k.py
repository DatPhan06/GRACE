import pandas as pd
import os

from utils.read_config import read_config

if __name__ == "__main__":
    config_value = read_config()
    k = 50
    n = 300
    file_path_output_model = config_value['insp_output_path']

    output = pd.read_csv(os.path.join(file_path_output_model, f'output_{n}.tsv'),
                         delimiter='\t',
                         names=['recall', 'id', 'target', 'summary', 'top_movie', 'candidate'])

    output = output[output['candidate'] != '[{}]']

    print(f'Recall@{k} = {sum(output.iloc[:,0]) / len(output)}')
 

    # combined_df = pd.concat(dfs, ignore_index=True)

    # # combined_df.columns = ['recall', 'recommend_movie_list', 'recommend_item', 'row']
    # print(sum(combined_df.iloc[:,0])/len(combined_df))
