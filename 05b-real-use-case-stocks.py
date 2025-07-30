import asyncio
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from rich.console import Console
from rich.prompt import Prompt
from src.yahoo_finance import YahooFinance

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

console = Console()

system_prompt = """
# **Step-by-Step Process for Generating the Report**

### **Step 1: Executive Summary**
- Summarize the company’s core business, industry, and competitive positioning.
- Provide an overview of the stock’s recent performance.
- Highlight key takeaways from the analysis.

### **Step 2: Market Research & Industry Analysis**
- Describe the company's industry, market size, and growth potential.
- Identify the company’s key competitors and compare their market positioning.
- Discuss macroeconomic factors impacting the industry (e.g., interest rates, inflation, regulations, geopolitical risks).

### **Step 3: Company Financial Analysis**
- Analyze revenue growth, profitability, and earnings trends over the last 3-5 years.
- Calculate and interpret key financial ratios:
  - **Profitability Metrics:** Gross Margin, Net Margin, Return on Equity (ROE).
  - **Liquidity Metrics:** Current Ratio, Quick Ratio.
  - **Leverage Metrics:** Debt-to-Equity Ratio, Interest Coverage Ratio.
  - **Valuation Metrics:** Price-to-Earnings (P/E), Price-to-Sales (P/S), Price-to-Book (P/B).
- Evaluate recent earnings reports, guidance, and management commentary.

### **Step 4: Market Trends & Stock Performance**
- Analyze recent stock performance, including price trends and trading volume.
- Compare the stock’s performance to industry benchmarks (e.g., S&P 500, Nasdaq, sector indices).
- Identify any major news or events that have influenced the stock price (e.g., earnings surprises, new product launches, regulatory approvals, acquisitions).

### **Step 5: Investment Recommendation (Buy, Sell, or Hold)**
- Provide a clear and well-supported investment recommendation.
- Highlight potential risks and catalysts that could impact future stock performance.
- Suggest an optimal entry and exit strategy based on technical and fundamental analysis.

## **Output Format**
The report should be structured as follows:

1. **Executive Summary** (Brief overview of the company and stock performance)
2. **Market Research & Industry Analysis** (Industry growth, competition, macro trends)
3. **Company Financial Analysis** (Revenue, margins, key financial ratios)
4. **Market Trends & Stock Performance** (Price movement, sector comparison, major events)
5. **Analyst Ratings & Institutional Sentiment** (Wall Street consensus, insider trading)
6. **Investment Recommendation** (Final verdict: Buy, Sell, or Hold + justifications)
"""

async def save_report_as_file(report_content: str, file_name: str) -> None:
    """
    Save the generated report content as a text file.
    Args:
        report_content (str): The content of the report.
        file_name (str): The name of the file to save the report.

    Returns:
        None
    """
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(report_content)
        return f"Report saved successfully as '{file_name}'"
    except Exception as e:
        return f"An error occurred while saving the report: {str(e)}"
    
model_settings = OpenAIModelSettings(
    temperature=0.2,
    max_tokens=8000
)

model = OpenAIModel('gpt-4o')

agent = Agent(
    name='Stock Researcher Agent',
    model=model,
    model_settings=model_settings,
    system_prompt=system_prompt,
    tools=[
        YahooFinance.get_current_price,
        YahooFinance.get_company_info,
        YahooFinance.get_historical_stock_prices,
        YahooFinance.get_stock_fundamentals,
        YahooFinance.get_income_statements,
        YahooFinance.get_key_financial_ratios,
        YahooFinance.get_analyst_recommendations,
        YahooFinance.get_company_news,
        YahooFinance.get_technical_indicators,
        save_report_as_file
    ]
)

@agent.system_prompt
def today_date() -> str:
    return f'Today is {datetime.today().strftime('%Y-%m-%d')}'

message_history = []

async def main():
    global message_history

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]")

        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold red]Exiting...[/bold red]")
            break

        console.print("[bold blue]Agent:[/bold blue] ", end="")  # Print "Agent: " once

        response_text = ""  # To store the full response for history

        async with agent.run_stream(user_input, message_history=message_history) as response:
            async for message in response.stream_text(delta=True):
                console.print(message, end="", style="bold blue", highlight=False)
                response_text += message  # Accumulate message for history

        console.print("\n")  # Ensure proper line break after streaming completes
        message_history.extend(response.new_messages())  # Update message history

asyncio.run(main())
