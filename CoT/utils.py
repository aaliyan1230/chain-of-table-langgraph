from typing import Annotated, Sequence
import pandas as pd
import pandas as pd


def load_files(path): #Annotated[Sequence[str], "Path of files, could be one or many"]) -> Annotated[Sequence[str], "List of dataframes"]:

    df_list = []
    for file_path in path:
        df_list.append(pd.read_csv(file_path, index_col=0).copy())

    # create df_dic for use by python eval() in evaluate_pandas_chain
    df_dic = {}
    for i, dataframe in enumerate(df_list):
        df_dic[f"df{i + 1}"] = dataframe

    return df_dic, df_list


# parser for action chain
def get_action(actions):
    if "<BEGIN>" in actions:
        a = actions.split('->')[1].strip()
    else:
        a = actions.split('->')[0].strip()
    return  a


def save_new_chain(user_query, output_chain, answer):
    try:
        df = pd.read_csv('./data/chains.csv', index_col=0)
        df.loc[len(df)] = {'query':user_query, 'chain':output_chain, 'answer': answer}
        df.to_csv('./data/chains.csv')

    except FileNotFoundError:
        df = pd.DataFrame([[user_query, output_chain, answer]])
        df.columns = ['query', 'chain', 'answer']
        df.to_csv('./data/chains.csv')


def get_last_chains(how_many=5):
    try:
        df = pd.read_csv('./data/chains.csv', index_col=0)
        return df.tail(how_many)
    except:
        return ''
