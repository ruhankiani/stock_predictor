# Stock Predictor

This project is a machine learning-based stock price predictor. The model predicts the next day's stock price based on the previous 100 days of data. It uses historical stock data fetched from the Alpha Vantage API and processes it for training and testing.

---

## Features

- **Stock Price Prediction**: Predicts the next day's stock price using the last 100 days of data.
- **Data Processing**: Processes historical stock data using rolling windows for inputs and outputs.
- **Model Training and Testing**: Trains on 70% of the data and tests on the remaining 30%.
- **Accurate Predictions**: Achieved high accuracy during testing.
- **Scalable Design**: Built using Python and TensorFlow.

---

## Data Source

The stock data is fetched from the [Alpha Vantage API](https://www.alphavantage.co/). To use this project, you will need a valid API key from Alpha Vantage.

---

## How It Works

1. **Data Collection**: 
   - The Alpha Vantage API is used to retrieve historical stock data.
   - Data is saved locally for reuse, with an option to refresh for the latest data.

2. **Data Processing**: 
   - Data is normalized using MinMaxScaler for better performance.
   - Inputs consist of the last 100 days, and the output is the stock price on the 101st day.

3. **Model Architecture**:
   
     - Input Layer: Accepts 100 days of stock prices.
     - Layer 1: 50 units, ReLU activation, with dropout (20%).
     - Layer 2: 60 units, ReLU activation, with dropout (30%).
     - Layer 3: 80 units, ReLU activation, with dropout (40%).
     - Layer 4: 120 units, ReLU activation, with dropout (50%). 
     - Output Layer: 1 unit, representing the predicted stock price.
   
   Loss Function: Mean Squared Error.
   
   Optimizer: Adam.

5. **Training and Testing**:
   - The model is trained on 70% of the data.
   - The remaining 30% is used for testing.
   - Loss function: Mean Squared Error.
   - Optimizer: Adam.

6. **Visualization**:
   - The predicted and actual stock prices are plotted to visualize performance.

---

## Requirements
- Get API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
- Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/stock-price-predictor.git
   cd stock-price-predictor
   ```
2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
3. **Add your Alpha Vantage API key to `prepare_data.py` line 82:**
  ```bash
  API_KEY = "your_alpha_vantage_api_key"
  ```
4. **Run the script:**
  ```bash
  python main.py
  ```

## Usage

1. **Input Stock Symbol**: Provide a valid stock symbol (e.g., `AAPL` for Apple).
2. **Data Selection**: Choose whether to use existing data or fetch updated data.
3. **Model Selection**: Opt to use a saved model or train a new one.
4. **Prediction**: View the predicted stock prices plotted against actual prices.

---

## Results

The Stock Predictor demonstrated high accuracy during testing. Predictions closely aligned with actual stock prices, as visualized in the generated graphs.

---

## Limitations

- The model's accuracy depends on the quality of the data from the Alpha Vantage API.
- Predictions may not fully account for unexpected market volatility or external factors.

---

## Future Enhancements

- Expand the model to predict stock prices for multiple future days.
- Integrate additional data sources, such as financial news and market indicators.
- Develop a web interface for easier usability.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **TensorFlow**: For providing the tools to build the neural network.
- **Alpha Vantage**: For supplying the stock price data.

For any questions or suggestions, please open an issue in the repository or contact the author.

