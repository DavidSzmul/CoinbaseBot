This project consists in creating a library in order to easily study the cryptocurrencies on Coinbase in order to train a IA Bot to determine h24/7d what is the best crypto
to invest on.

<img src="./assets/bot.PNG" width="240">

## Extraction of Cryptocurrency historic
Use of Scrapping + Dataframe (Python) to generate the database used for train/test

## Training of Bot
Use of Reinforcement Learning principle -> Agent + Environment. Implementation of environment specifically for trading, whatever the number of trade studied

## Testing of Bot
Simulation of the behavior of the bot (on untrained database) to guarantee that Bot is coherent and at least does not make us loose money

## Coinbase Transaction
Coinbase enables conversion from one crypto to another very easily. However this is not doable using their REST API (unfortunately). 
The solution is to use scrapping again to reproduce mouse and keyboard on browser to realize a transaction, based on Bot decision.

## Front-End
Once the Bot library is done (Back-End), the idea would be to integrate an IHM to visualize Bot behavior (train/test/real-time).
This is firstly done using Tkinter + Matplotlib libraries.
For perspectives, this could be implemented using a Client-Server App with a Front-End done in HTML/JS (preffered framework would be Vue)




## Authors
David Szmul - [Github](https://github.com/DavidSzmul) | [LinkedIn](https://www.linkedin.com/in/david-szmul-207564134/)   
`Follow me if you are interested in my project and you would like to contribute !`
