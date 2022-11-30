# Forecast integrated TAF environment (FITE)



*GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset[1] of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data.*
> ✨ Bootstrapped with Create Snowpack App (CSA).

## Model: Training
``` bash
python -m model train
***** Tokenizer not found, creating tokenizer... *****
***** Model not found, creating model... *****
***** Dataset not found, creating dataset... *****
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2433/2433 [00:05<00:00, 417.94ba/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 609/609 [00:01<00:00, 450.50ba/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2433/2433 [00:07<00:00, 340.06ba/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 609/609 [00:01<00:00, 350.89ba/s]
***** Loading dataset... *****
```
The model was trained on ~90-100 TAFs from each of the following locations:

``` bash
[
    "KADW" 
    "KBLV"
    "KDAA"
    "KDOV"
    "KFAF"
    "KFFO"
    "KFTK"
    "KGUS"
    "KHOP"
    "KLFI"
    "KMTC"
    "KMUI"
    "KOFF"
    "KRDR"
    "KWRI"
    "PABI"
    "PAFB"
    "PASY"
]
```

any TAF that had a COR remark was removed from the dataset
### limitations:

- TAF data does not include year and month in the body of the text.  So there is no way for the model to learn seasonal patterns like lake effect snow or thunderstorms in the summer.  This could be solved by adding the year and month to the TAF body text.

- The model reads left to right and with the TX/TN groups at the end of the forecast, the model is not able to infer precipitation types or icing conditions.  This could be solved by moving the TX/TN groups to the beginning of the forecast.

## API: Usage
once the model has been trained you can access the TAF pipeline via the api module
```  python
Python 3.10.6 (main, Nov  2 2022, 18:53:38) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from api.main import Engine
>>> taf_pipeline = Engine["TAF"]
>>> taf_pipeline.generate_forecast("TAF KBLV 181730Z 1818/1918 VRB06KT") # chooses a random preset hyper-parameter strategy
[['TAF KBLV 181730Z 1818/1918 VRB06KT 9999 FEW010 SCT200 QNH2997INS', 'BECMG 1800/1801 VRB06KT 9999 SCT020 OVC200 QNH2999INS', 'BECMG 1811/1812 VRB06KT 9999 SCT030 BKN200 QNH3001INS', 'BECMG 1814/1815 26015KT 9999 SCT030 BKN200 QNH3005INS TX11/1719Z TN00/1810Z']]
>>> taf_pipeline.generate_forecast("TAF KBLV 181730Z 1818/1918 VRB06KT", "GREEDY")
[['TAF KBLV 181730Z 1818/1918 VRB06KT 9999 FEW050 QNH3029INS', 'BECMG 1714/1715 26010G15KT 9999 FEW060 QNH3028INS TX02/1620Z TNM03/1712Z']]
>>> taf_pipeline.generate_forecast("TAF KBLV 181730Z 1818/1918 VRB06KT", "TOP_K5")
[['TAF KBLV 181730Z 1818/1918 VRB06KT 9999 FEW020 BKN040 QNH2993INS', 'BECMG 1702/1703 27012KT 9999 FEW030 BKN250 QNH3010INS', 'BECMG 1710/1711 VRB06KT 9999 SCT250 QNH3020INS TX05/1620Z TN00/1712Z']]
>>> results, = taf_pipeline.generate_forecast("TAF KBLV 181730Z 1818/1918 27015G30KT 8000", "TOP_K5")
>>> print("\n".join(results))
TAF KBLV 181730Z 1818/1918 27015G30KT 8000 -SN OVC015 620156 QNH3005INS
BECMG 1711/1712 27009KT 9999 NSW BKN025 620257 QNH3010INS TXM04/1619Z TNM13/1710Z
```

or load the model and tokenizer directly from the store via the transformers API, the pipeline provides a more convenient interface and preset configuration


``` python
>>> from transformers import GPT2LMHeadModel, GPT2TokenizerFast
>>> model = GPT2LMHeadModel.from_pretrained("store/models/gpt2-taf-0.1.0")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("store/tokenizer/gpt2-taf-0.1.0")
```


### TODO:

- [ ] add year and month to TAF body text
- [ ] move TX/TN groups to beginning of forecast
- [ ] add more locations to dataset
- [ ] add more TAFs from each location to dataset


### API: Scripts
``` bash
npm run api

> api
> uvicorn api.main:app --reload

INFO:     Will watch for changes in these directories: ['/home/leaver2000/fite-gpt2']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [19228] using StatReload
^BINFO:     Started server process [19230]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     127.0.0.1:44398 - "GET / HTTP/1.1" 200 OK
```

starts the FastAPI Uvicorn server

### CLIENT: Available Scripts

### npm start

Runs the app in the development mode.
Open <http://localhost:8080> to view it in the browser.

The page will reload if you make edits.
You will also see any lint errors in the console.

### npm run build

Builds a static copy of your site to the `build/` folder.
Your app is ready to be deployed!

**For the best production performance:** Add a build bundler plugin like "@snowpack/plugin-webpack" to your `snowpack.config.mjs` config file.



#### routes
<!-- POST ROUTE -->
``` bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate/TAF/GREEDY' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "userInput": "TAF KBLV 181730Z 1818/1918 "
}'
["TAF KBLV 181730Z 1818/1918 VRB06KT 9999 FEW050 QNH3029INS","BECMG 1714/1715 26010G15KT 9999 FEW060 QNH3028INS TX02/1620Z TNM03/1712Z"]
```


## Environment

``` bash
python -m torch.utils.collect_env

Collecting environment information...
PyTorch version: 1.13.0+cu117
Is debug build: False
CUDA used to build PyTorch: 11.7
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.1 LTS (x86_64)
GCC version: (Ubuntu 11.2.0-19ubuntu1) 11.2.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.6 (main, Nov  2 2022, 18:53:38) [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-5.10.102.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.5.119
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 2080 SUPER
Nvidia driver version: 516.59
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.2.4
/usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.2.4
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] mypy-extensions==0.4.3
[pip3] numpy==1.23.5
[pip3] torch==1.13.0+cu117
[conda] Could not collect
```
### install



## client

![image](https://user-images.githubusercontent.com/76945789/203187237-31a110a3-c340-4995-a24a-8be634f8c587.png)

## server
![image](https://user-images.githubusercontent.com/76945789/203183599-ba4adad0-d87b-407a-94ac-d9acb2c19d08.png)


## references
[gpt and casual language models](https://huggingface.co/transformers/v2.0.0/examples.html#gpt-2-gpt-and-causal-language-modeling)