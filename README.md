# RDN-with-Pruning

  
* Docker Image
<pre>
<code>
$ sudo docker pull heejowoo/pruning_rdn:0.5
</code>
</pre>

* Super Resolutino Deep Network 중 하나인 RDN에 대해 기 학습된 모델을 이용하여 0.5 Pruning Ratio로 설정하여 반복적인 Pruning과 재학습 진행
|x4, Pruning Ratio : 0.5|No Pruning(PSNR(dB))|Apply pruning once(PSNR(dB))|Iterative Pruning & Retrain(PSNR(dB))|
|-----------------------|--------------------|----------------------------|-------------------------------------|
|Set5|32.4|25.64|32.41|
|Set14|28.78|24.14|28.82|
|BSD100|27.71|23.99|27.74|
|Urban100|26.40|22.60|26.48|

