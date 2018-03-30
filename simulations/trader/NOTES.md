# Crypto currency trader NEAT

Implementación de NEAT para trading de criptomonedas, usando otto trader.

## Notas
* Realizar evaluación de cada individuo de cada generación con el histórico total de las monedas, como si fuera el "modelo" a evaluar
* Variables de entrada de la red
* Medir rendimiento al final de la simulación con el histórico y ese será el fitness
* Medir la ganancia/pérdida con la variación del siguiente punto de datos
* Considerar la pérdida por comisión del trade


## Concepts
* Fitness = Profitability
* Model = Historic data
* 

## Resources

Traiding data and conversion to USD: https://cdn.patricktriest.com/blog/images/posts/crypto-markets/Cryptocurrency-Pricing-Analysis.html 


# Get coin prices
* https://www.coindesk.com/price/ - request the period and interval
  * https://api.coindesk.com/charts/data?data=ohlc&startdate=2017-12-28&enddate=2018-01-04&exchanges=bpi&dev=1&index=USD

* https://www.cryptocompare.com/api/#-api-data-histohour-
  * https://min-api.cryptocompare.com/data/histohour?fsym=ETH&tsym=USD&limit=60&aggregate=3&e=Kraken&extraParams=your_app_name 
  * https://min-api.cryptocompare.com/data/histohour?fsym=BTC&tsym=USD&limit=300&aggregate=1&e=CCCAGG
  * Ripple to USD hourly
    * https://min-api.cryptocompare.com/data/histohour?fsym=XRP&tsym=USD&limit=300&aggregate=1&e=CCCAGG
  * ETH to USD hourly
    * https://min-api.cryptocompare.com/data/histohour?fsym=ETH&tsym=USD&limit=300&aggregate=1&e=CCCAGG


# Coin price history algorithm
- Get all coins vs Bitcoin Price
- Get bitcoin price vs USD
- Get bitcoin price per hour and convert the coins price to USD
