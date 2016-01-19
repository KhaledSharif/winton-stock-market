# Winton Stock Market Challenge

This repository will contain my ongoing work in the Winton Stock Market Challenge. In the competition, the challenge is to predict the return of a stock, given the history of the past few days. It provides a good machine learning challenge, due to the very large amount of noisy, un-stationary data provided with ~150 features to predict ~60 regression targets, given weights for evaluation. The image below, credit to Kaggle and Winton, explains briefly the data and objectives.

<img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/4504/media/Presentation1%20(1).jpg" />
<i>Infographic from the Kaggle competition that details the given data format</i>


<img src="http://i.imgur.com/fXqkKwf.png" />
<i>The result of my deep neural network approach to the problem, using <a href="https://github.com/KhaledSharif/winton-stock-market/blob/master/highway-network.py">a 4 layer highway network</a>.</i>


<img src="http://i.imgur.com/icKtNEm.png" />
<i>The result of running the exact same network previously, but using the Adamax update function instead of the Adadelta function. The graph shows a much smoother decent towards the minimum.</i>
