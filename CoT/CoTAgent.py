# basics
import json
import pandas as pd
import traceback
from utils import get_action, get_last_chains

# pydantic
from typing import Annotated, Sequence

# langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import FunctionMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# langgraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END

class CoTAgent:

    def __init__(self, df_dic, df_list, system_prompt, state) -> None:
        self.df_dic: list = df_dic
        self.df_list: list = df_list
        self.system_prompt = system_prompt
        self.state = state
        self.tool_executor = None
        self.model = None

    # function to evaluate the next action in the chain
    @tool
    def evaluate_pandas_chain(self,
        chain: Annotated[Sequence[str], "The pandas chain of actions . e.g. df1.groupby('age').mean() -> df1.sort_values() -> <END>"],
        inter=None):
        """Use this to execute the chain of pandas code on the dataframes"""

        name = "evaluate_pandas_chain"

        try:
            action = get_action(chain)#.replace(toreplace, 'inter')
            print('\n\naction: ', action)
            
            inter = eval(action, {"inter": inter, "df_dic": self.df_dic})
            
            if isinstance(inter, pd.DataFrame):
                intermediate = inter.head(50).to_markdown()
            else:
                intermediate = inter

            return intermediate, action, inter
                    
        except Exception as e:
            return f"An exception occured: {traceback.format_exc()}", action, None

    # function to look at dataframes 
    @tool
    def view_pandas_dataframes(self,
        df_list: Annotated[Sequence[str], "List of maximum 3 pandas dataframes you want to look at, e.g. [df1, df2, df3]"]):
        """Use this to view the head(10) of dataframes to answer your question"""

        name = "view_pandas_dataframes"

        markdown_str = "Here are .head(10) of the dataframes you requested to see:\n"
        for df in df_list:
            df_head = self.df_dic[df].head(10).to_markdown()
            markdown_str += f"{df}:\n{df_head}\n"

        markdown_str = markdown_str.strip()
        return markdown_str
    
    def get_tools(self):
        tools = [self.evaluate_pandas_chain, self.view_pandas_dataframes]
        if not self.tool_executor:
            self.tool_executor = ToolExecutor(tools)
        return tools
    
    @staticmethod
    def get_chain_examples() -> str:
        chain_examples = ""
        if(type(get_last_chains()) == pd.core.frame.DataFrame):
            for index, row in get_last_chains()[["query", "chain"]].iterrows():
                chain_examples += f'Question: {row["query"]}\nChain: {row["chain"]}\n\n'
        return chain_examples

    def structured_base_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_prompt,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(num_dfs=len(self.df_list))
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in self.get_tools()]))
        # prompt = prompt.partial(questions_str=questions_str)

        # passing in past successful queries
        chain_examples = ""
        if(type(get_last_chains()) == pd.core.frame.DataFrame):
            for index, row in get_last_chains()[["query", "chain"]].iterrows():
                chain_examples += f'Question: {row["query"]}\nChain: {row["chain"]}\n\n'
        prompt = prompt.partial(chain_examples=chain_examples)

        return prompt
    
    # Define the function that determines whether to continue or not: conditional edge
    def should_continue(self, state):
        messages = state['messages']
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # Otherwise if there is, we continue with the call_tool node
        else:
            return "continue"

    # Define the function that calls the model
    def call_model(self, state):
        
        response = self.model.invoke(state)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the function to execute tools
    def call_tool(self, state):
        messages = state['messages']
        
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        
        
        tool_input = last_message.additional_kwargs["function_call"]["arguments"]
        

        tool_input_dict = json.loads(tool_input)
        tool_input_dict['inter'] = state['inter']

        if last_message.additional_kwargs["function_call"]["name"] == 'view_pandas_dataframes':
            
            # We construct an ToolInvocation from the function_call
            action = ToolInvocation(
                tool=last_message.additional_kwargs["function_call"]["name"],
                tool_input=tool_input_dict,
            )
            # We call the tool_executor and get back a response
            response = self.tool_executor.invoke(action)

            function_message = FunctionMessage(content=str(response), name=action.tool)
            return {"messages": [function_message]} # ,"actions": [attempted_action]}
        

        # if the tool is to evaluate chain the chain
        elif last_message.additional_kwargs["function_call"]["name"] == 'evaluate_pandas_chain':
    
            # We construct an ToolInvocation from the function_call
            action = ToolInvocation(
                tool=last_message.additional_kwargs["function_call"]["name"],
                tool_input=tool_input_dict,
            )
            # We call the tool_executor and get back a response
            response, attempted_action, inter = self.tool_executor.invoke(action)
                
            if "An exception occured:" in str(response):
                error_info = f"""
                You have previously performed the actions: 
                {state['actions']}

                Current action: 
                {attempted_action}

                Result .head(50): 
                {response}

                You must correct your approach and continue until you can answer the question:
                {state['question']}

                Continue the chain with the following format: action_i -> action_i+1 ... -> <END>
                """
                print(error_info)

                function_message = FunctionMessage(content=str(error_info), name=action.tool)
                return {"messages": [function_message]}
            
            else:

                success_info = f"""
                You have previously performed the actions: 
                {state['actions']}

                Current action: 
                {attempted_action}

                Result .head(50):
                {response}

                You must continue until you can answer the question:
                {state['question']}

                Continue the  chain with the following format: action_i -> action_i+1 ... -> <END>
                """
                print(success_info)

                # We use the response to create a FunctionMessage
                function_message = FunctionMessage(content=str(success_info), name=action.tool)
                # We return a list, because this will get added to the existing list
                return {"messages": [function_message], "actions": [attempted_action], "inter": inter}
    
    def instanciate_ai_model(self):
        prompt = self.structured_base_prompt()
        tool_executor = self.tool_executor

        functions = [convert_to_openai_function(t) for t in self.get_tools()]

        model = prompt | ChatOpenAI(model="gpt-4-0125-preview").bind_functions(functions)

        if not self.model:
            self.model= model

    def compile_graph(self):
        # Define a new graph
        workflow = StateGraph(self.state)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.call_model)
        workflow.add_node("action", self.call_tool)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "action",
                # Otherwise we finish.
                "end": END
            }
        )
        print("passed________________________-")

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge('action', 'agent')

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        return workflow.compile()
        

