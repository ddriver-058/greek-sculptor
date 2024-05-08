## Demo & Summary
[DEMO](https://greek-sculptor.scaeangate.io)

GreekSculptor is an application built in Python Dash, backed by a Flask API, that finds optimal selections from the option chain of a chosen US stock symbol. Users can input their optimization criteria, (e.g., maximize vega such that delta = 0), and then the optimizer will return a selection vector specifying the portfolio of options satisfying the objective and constraints.

## Repository Structure
The repository consists of code (in the api and app directories) and Helm chart deployments (in the api-chart and app-chart folders).

While this is a public branch with generic values, in my personal setup, ArgoCD manages deployments with separate dev and production branches, enabling isolated testing and automated production updates.

## Additional Context
This tool was built after some initial experiments I made to apply linear programming to options trading in R. I thought the results were interesting enough that I wanted to share the technique by building an application that is easy to use and lets users quickly inspect the optimization results.

As the "About this App" page in GreekSculptor mentions, this app is for educational and entertainment purposes only. It does not constitute financial advice.
