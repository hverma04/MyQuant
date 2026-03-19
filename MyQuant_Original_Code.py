import yfinance as yf
import pandas as pd
import numpy as np
import numpy_financial as npf
from scipy.stats import norm
import matplotlib.pyplot as plt

def get_options_chain(ticker_symbol):
    while True:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        
        if not expirations:
            print(f"No options data found for {ticker_symbol}")
            ticker_symbol = input("Enter stock ticker: ").upper()
            continue

        print(f"\nAvailable expiration dates for {ticker_symbol}:")
        for i, exp in enumerate(expirations, start=1):
            print(f"{i}. {exp}")

        try:
            choice = int(input("\nSelect expiration number: "))
            selected_exp = expirations[choice - 1]
        except (ValueError, IndexError):
            print("Invalid selection. Try again.")
            continue

        option_chain = ticker.option_chain(selected_exp)
        return ticker_symbol, ticker, selected_exp, option_chain.calls, option_chain.puts

def select_option(ticker_symbol, calls, puts):
    trade_type = input("Enter trade type (call/put): ").lower()
    if trade_type == "call":
        available_strikes = calls["strike"].tolist()
        print(f"\nAvailable Strike Prices for {ticker_symbol} Calls:\n{available_strikes}")
    else:
        available_strikes = puts["strike"].tolist()
        print(f"\nAvailable Strike Prices for {ticker_symbol} Puts:\n{available_strikes}")

    while True:
        try:
            strike_price = float(input("Enter Strike Price: "))
            option_row = calls[calls["strike"] == strike_price] if trade_type == "call" else puts[puts["strike"] == strike_price]
            if option_row.empty:
                print(f"Strike {strike_price} not found. Available: {available_strikes}")
                continue
            break
        except ValueError:
            print("Invalid numeric input.")

    return option_row, strike_price, trade_type

def percent_and_breakeven(option_row, ticker, ticker_symbol, trade_type):
    current_price = ticker.history(period="1d")["Close"].iloc[0]
    strike_price = option_row["strike"].iloc[0]
    premium = option_row["ask"].iloc[0]
    
    while True:
        try:
            target_price = float(input(f"Enter {ticker_symbol} Target Price: "))
            break
        except ValueError:
            print("Invalid Input. Enter valid Price Input.")

    breakeven_price = strike_price + premium if trade_type == "call" else strike_price - premium
    target_percent_move = ((target_price - current_price) / current_price) * 100
    percent_move_strike = ((strike_price - current_price) / current_price) * 100
    breakeven_percent_move = ((breakeven_price - current_price) / current_price) * 100

    print(f"\nTicker: {ticker_symbol}\nTo Target Price Move: {target_percent_move:.2f}%\nTo Strike Price Move: {percent_move_strike:.2f}%\nTo Breakeven Price Move: {breakeven_percent_move:.2f}%\nBreakeven Stock Price: ${breakeven_price:.2f}")

    return {
        "ticker_symbol": ticker_symbol, "current_price": current_price, "target_price": target_price,
        "premium": premium, "strike_price": strike_price, "trade_type": trade_type,
        "breakeven_price": breakeven_price, "Implied_Volatility": option_row["impliedVolatility"].iloc[0]
    }

def PnL_calculator(base_calculations):
    strike_price = base_calculations["strike_price"]
    premium = base_calculations["premium"]
    trade_type = base_calculations["trade_type"]
    target_price = base_calculations["target_price"]

    while True:
        try:
            order_size_input = int(input(f"Enter Order Size (# of {trade_type.capitalize()} contracts): "))
            order_size = order_size_input * 100
            break
        except ValueError:
            print("Invalid Input.")

    intrinsic_value = max(0, target_price - strike_price) if trade_type == "call" else max(0, strike_price - target_price)
    profit_loss = intrinsic_value - premium
    return_ratio = profit_loss / premium 
    
    print(f"Intrinsic Value: ${intrinsic_value:.2f}\nP&L per Contract: ${profit_loss:.2f}\nTotal P&L: ${profit_loss * order_size:.2f}\nReturn Ratio: {return_ratio:.2f}\nPremium Paid: ${premium:.2f}")
    return {"per_contract": profit_loss, "total": profit_loss * order_size, "return_ratio": return_ratio, "order_size": order_size}

def expected_move(base_calculations, selected_exp):
    current_price = base_calculations["current_price"]
    annual_iv = base_calculations["Implied_Volatility"]
    time_to_exp = (pd.to_datetime(selected_exp) - pd.to_datetime("today")).days / 365
    exp_move = current_price * annual_iv * np.sqrt(time_to_exp)
    
    upper_bound = current_price + exp_move
    lower_bound = current_price - exp_move

    print(f"\nExpected Price Range: ${lower_bound:.2f} - ${upper_bound:.2f}")
    return {"periodic_IV": annual_iv * np.sqrt(time_to_exp), "time_to_exp": time_to_exp, "upper_bound": upper_bound, "lower_bound": lower_bound}

def z_score(base_calculations, expected_move_data):
    current_price = base_calculations["current_price"]
    O = expected_move_data["periodic_IV"]
    return {
        "Target_Z": np.log(base_calculations["target_price"] / current_price) / O,
        "Strike_Z": np.log(base_calculations["strike_price"] / current_price) / O,
        "Breakeven_Z": np.log(base_calculations["breakeven_price"] / current_price) / O
    }

def win_lose(z_scores, base_calculations):
    t_win = 1 - norm.cdf(z_scores["Target_Z"]) if base_calculations["trade_type"] == "call" else norm.cdf(z_scores["Target_Z"])
    s_win = 1 - norm.cdf(z_scores["Strike_Z"]) if base_calculations["trade_type"] == "call" else norm.cdf(z_scores["Strike_Z"])
    b_win = 1 - norm.cdf(z_scores["Breakeven_Z"]) if base_calculations["trade_type"] == "call" else norm.cdf(z_scores["Breakeven_Z"])

    print(f"\nTarget Prob: {t_win:.2%}\nStrike Prob: {s_win:.2%}\nBreakeven Prob: {b_win:.2%}")
    return {"Breakeven_Win": b_win, "Breakeven_Loss": 1 - b_win, "Target_Win": t_win, "Strike_Win": s_win}

def expected_value(base_calculations, win_lose_data, pnl):
    while True:
        try:
            sl = float(input("Enter Stop Loss % (0-100): ")) / 100
            break
        except ValueError: print("Invalid.")
    
    ev = (win_lose_data["Breakeven_Win"] * pnl["total"]) - (win_lose_data["Breakeven_Loss"] * (base_calculations["premium"] * pnl["order_size"]) * sl)
    print(f"Expected Value: ${ev:.2f}")
    return ev

def monte_carlo_simulation(base_calculations, expected_move_data):
    prices = np.random.lognormal(mean=np.log(base_calculations["current_price"]), sigma=expected_move_data["periodic_IV"], size=10000)
    plt.hist(prices, bins=50, edgecolor='k', alpha=0.6)
    plt.axvline(base_calculations["current_price"], color='blue', label='Current')
    plt.title(f'Price Distribution for {base_calculations["ticker_symbol"]}')
    plt.legend()
    plt.show()

def main():
    while True:
        ticker_input = input("\nEnter stock ticker: ").upper()
        ticker_symbol, ticker, selected_exp, calls, puts = get_options_chain(ticker_input)
        option_row, strike_price, trade_type = select_option(ticker_symbol, calls, puts)
        base_calc = percent_and_breakeven(option_row, ticker, ticker_symbol, trade_type)
        pnl = PnL_calculator(base_calc)
        exp_data = expected_move(base_calc, selected_exp)
        z_sc = z_score(base_calc, exp_data)
        wl_data = win_lose(z_sc, base_calc)
        ev_val = expected_value(base_calc, wl_data, pnl)
        
        monte_carlo_simulation(base_calc, exp_data)
        
        rerun = input("\nRun another analysis? (Y/N): ").strip().lower()
        if rerun != 'y':
            break

if __name__ == "__main__":
    main()