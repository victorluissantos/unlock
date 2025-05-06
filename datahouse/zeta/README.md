# Zeta

Get Started, you need go python caontainer:
`docker-compose exec python bash`

And than, inside zeta folder:
`cd zeta`

We'll go run trainners:

Question:
How run?
Answer:
`python3 main.py https://exemplo.com/captcha 100 captchas`
If the origin of URL return a base64 need use:
`python main.py https://www.exemplo.com/detranextratos/rest/captcha/req 100 captchas base64`


Is a normal type of captcha:

- Before processing:


- After processing:


## Results
Acurácia do modelo: 58.82%
Modelo salvo em: modelo_svm.joblib