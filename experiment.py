from models import pen_inv_model
from scraper import dataset
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def experiment(stocks, periods):

  for stock in stocks.split():
    print(f"\nRESULTS FOR {stock}\n")
    RMSE_scores = {}
    MAPE_scores = {}

    for period in periods:
      print("PERIOD:"+period)
      d = dataset(stock,period)
      model = pen_inv_model(d)
      model.train()
      y_prediction = model.f(d.X_te)

      RMSE_scores[period] = model.RMSE()
      MAPE_scores[period] = model.MAPE()

      # plot predicted against input with actual
      pred, = plt.plot(d.X_te[:,1:], y_prediction, c='crimson')
      pred.set_label('Line of regression')
      actu= plt.scatter(d.X_te[:,1:], d.Y_te, c ='blue')
      actu.set_label('Actual Closing Prices')
      plt.legend()
      plt.xlabel('Opening Price')
      plt.ylabel('Closing Price')
      plt.tight_layout()
      plt.show()

    # d is max period dataset
    plot_acf(d.X_te[:,1:], lags=40)
    plt.show()

    plt.plot(list(RMSE_scores.keys()),list(RMSE_scores.values()))
    plt.grid()
    plt.title(f"RMSE error for {stock}")
    plt.show()

    plt.plot(list(MAPE_scores.keys()),list(MAPE_scores.values()))
    plt.grid()
    plt.title(f"MAPE error for {stock}")
    plt.show()

# valid periods: 5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
periods = ("5d","1mo","3mo","6mo","1y","2y","5y","10y","max")
experiment("BP AAPL MSFT",periods)