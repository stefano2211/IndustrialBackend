from app.agent.finance.graph import finance_graph
from langchain_core.messages import HumanMessage
import json

def test_finance_subgraph():
    print("Testing Finance Subgraph with AI Node...")
    query = "What is the price of Apple?"
    print(f"Query: '{query}'")
    
    try:
        result = finance_graph.invoke({"messages": [HumanMessage(content=query)]})
        stock_data = result.get("stock_data")
        ticker = result.get("ticker")
        
        print(f"Extracted Ticker: {ticker}")
        
        if stock_data:
            print("Success! Data received:")
            print(json.dumps(stock_data, indent=2))
        else:
            print("Failed: No stock data returned.")
            print("Result:", result)
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_finance_subgraph()
