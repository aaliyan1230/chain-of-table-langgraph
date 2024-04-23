import os
import pandas as pd
from utils import load_files
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from CoTAgent import (
    CoTAgent
)
from prompts import SYSTEM_PROMPT


file_paths = input("Enter space separated csv, xlsx file paths: \n").split()
paths = [path.strip() for path in file_paths]

paths=[
        "./data/coworker0.csv",
        "./data/coworker1.csv"
]

df_dic, df_list = load_files(
    path=paths
)

print(df_dic)

class CoTAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    actions: Annotated[Sequence[str], operator.add]
    inter: pd.DataFrame
    question: str
    memory: str

cot_agent = CoTAgent(df_dic=df_dic, df_list=df_list, system_prompt=SYSTEM_PROMPT, state=CoTAgentState)

cot_agent.instanciate_ai_model()

app = cot_agent.compile_graph()


user_query = input("Enter your query: ")
user_query = "Return how many times Steven Rollins was selected for MVP. Return also the number of grades this employee received for each MVP reason."


inputs = {"messages": [HumanMessage(content=user_query)], "actions":["<BEGIN>"], "question": user_query, "memory": ""}

for output in app.stream(inputs, {"recursion_limit": 40}):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        if key == "agent":
            print("🤖 Agent working...")
        elif key == "action":
            print("abc")
            if value["messages"][0].name == "view_pandas_dataframes":
                print("🛠️ Current action:")
                print("`viewing dataframes`")
            else:
                if "actions" in value.keys():
                    print(f"🛠️ Current action:")
                    print(f"`{value['actions']}`")
                    print(f"Current output:")
                    print(value["inter"])
                else:
                    print(f"⚠️ An error occured, retrying...")
        else:
            print("🏁 Finishing up...")
            print(f"Final output:")
            print(value["inter"])
            print(f"Final action chain:")
            print(" -> ".join(value["actions"])  + ' -> <END>')

        print("---")
        


output_dict = output["__end__"]
agent_response = output_dict["messages"][-1].content
final_table = output_dict["inter"]
final_message = agent_response.replace('<END>', '')

print(final_message)
print(final_table)

