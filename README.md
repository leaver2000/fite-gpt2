# Forecast integrated TAF environment (FITE)

## description

*GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset[1] of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data.*

The training scripts run based on the pyproject.toml config file

in you pyproject.toml 

``` toml
[tool.fite]
root-path = "store/" # the root path to the models and data 

[[tool.fite.models]]
model-name = "gpt2-taf-base1" # store/gpt2-taf-base1/[training-data.txt | training-data.json]

```


## install

``` bash
git clone ... && cd ...
python -m venv ~/venv
source ~/venv/bin/activate
pip install .
Processing /home/leaver2000/fite-gpt2
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  ...
```

## train the model

```bash
python -m fite.train --verbose taf
***** Training model: gpt2-taf-base2 *****
***** From the base model: gpt2 *****
***** Using dataset: store/gpt2-taf-base2/dataset *****
***** Tokenizer not found, creating tokenizer... *****
***** Model not found, creating model... *****
***** Dataset not found, creating dataset... *****
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2433/2433 [00:05<00:00, 417.94ba/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 609/609 [00:01<00:00, 450.50ba/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2433/2433 [00:07<00:00, 340.06ba/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 609/609 [00:01<00:00, 350.89ba/s]
***** Loading dataset... *****
...

[['TAF KBLV 020600 0200/0306 18010KT 8000 -SHRA OVC020 QNH2995INS', 'TEMPO 0200/0206 5000 -SHRA OVC015', 'BECMG 0207/0208 VRB06KT 0800 -SHRA BR OVC006 QNH2990INS', 'BECMG 0211/0212 VRB06KT 1600 BR OVC006 QNH2990INS TX20/0120Z TN11/0212Z'], ['TAF KGTB 251700Z 2517/2623 26012G20KT 9999 BKN030 QNH2993INS', 'BECMG 2601/2602 19006KT 8000 -SHRA BKN030 QNH2993INS', 'BECMG 2603/2604 18006KT 6000 -SHRA BKN030 QNH2993INS TX21/2519Z TN13/2610Z'], ['TAF KGTB 251700Z 2517/2623 26012G20KT 9999 OVC008 QNH2970INS', 'BECMG 2519/2520 27009KT 9999 SCT009 OVC015 QNH2976INS', 'BECMG 2610/2611 VRB06KT 9999 BKN015 OVC025 QNH2991INS', 'BECMG 2613/2614 29009KT 9999 SCT025 BKN040 QNH2994INS', 'BECMG 2616/2617 31010G15KT 9999 FEW050 SCT200 QNH3010INS TX19/2519Z TN11/2610Z'], ['TAF KMTC 252000Z 2520/2702 29012G20KT 9999 SKC QNH3030INS', 'BECMG 2601/2602 19006KT 9999 SKC QNH3030INS TXM06/2619Z TNM12/2610Z'], ['TAF PASY 251400Z 2514/2620 11006KT 9999 FEW030 FEW045 SCT100 QNH3002INS', 'BECMG 2519/2520 12006KT 9999 SCT030 BKN200 QNH3010INS', 'BECMG 2601/2602 19006KT 8000 -SHRA OVC020 510204 QNH3010INS', 'BECMG 2603/2604 18006KT 6000 -DZ BR OVC006 QNH3012INS TX20/2519Z TN13/2610Z']]
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
>>> from fite.api import PipelineEngine
>>> pipeline = PipelineEngine.get_pipeline("taf")
>>> pipeline.generate("TAF KBLV 010000Z 0100/0206", "GREEDY")
[['TAF KBLV 010000Z 0100/0206 VRB06KT 9999 BKN020 QNH3021INS', 'BECMG 0111/0112 VRB06KT 9999 BKN020 QNH3021INS', 'BECMG 0116/0117 VRB06KT 9999 BKN020 QNH3021INS TX20/0118Z TN11/0112Z']]
>>> pipeline.generate("TAF KBLV 010000Z 0100/0206 8000 -TSRA", "GREEDY")
[['TAF KBLV 010000Z 0100/0206 8000 -TSRA BKN010 OVC020 QNH2993INS', 'BECMG 0111/0112 VRB06KT 9999 NSW FEW015 FEW025 BKN040 QNH2992INS', 'BECMG 0116/0117 VRB06KT 9999 FEW200 QNH2992INS TX20/0120Z TN09/0112Z']]
```

or load the model and tokenizer directly from the store via the transformers API, the pipeline provides a more convenient interface and preset configuration


``` python
>>> from transformers import GPT2LMHeadModel, GPT2TokenizerFast
>>> model = GPT2LMHeadModel.from_pretrained(PATH_TO_TAF_MODEL)
>>> tokenizer = GPT2TokenizerFast.from_pretrained(PATH_TO_TAF_TOKENIZER>)
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
  'http://127.0.0.1:8000/generate/taf?strategy=TOP_P98' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": [
    "TAF KBLV 010000Z 0103/0112 00000KT 9999 SCT250",
    "TAF KDAA 010000Z 0103/0112 00000KT 5000"
  ]
}'
[["TAF KBLV 010000Z 0103/0112 00000KT 9999 SCT250 QNH3038INS","BECMG 0214/0215 09012KT 9999 SKC QNH3046INS","BECMG 0217/0218 04009KT 9999 BKN030 QNH3044INS","BECMG 0220/0221 05008KT 9999 SKC QNH3044INS","BECMG 0223/0604 VRB03KT 9999 SKC QNH3038INS TX21/0120Z TN09/0212Z"],["TAF KDAA 010000Z 0103/0112 00000KT 5000 BR OVC007 QNH2991INS TX20/0218Z TN04/0111Z"]]%  
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

### FastAPI swagger-ui
![swagger-ui](https://user-images.githubusercontent.com/76945789/204947048-046c6c0f-1dbf-4c81-b608-45aaac053097.png)
#### strategy

The Strategy enum is used to select the sampling strategy for the model. The default is GREEDY, which selects the highest probability token at each step. TOP_K selects the top k tokens at each step, where k is the value of the k parameter. TOP_P selects the smallest set of tokens whose cumulative probability exceeds the value of the p parameter. TOP_P98 is a shortcut for TOP_P with p=0.98. TOP_P is recommended for generating text, while TOP_K is recommended for generating code.


![image](https://user-images.githubusercontent.com/76945789/204947737-fc0be305-76a6-4d72-9215-e36eba6c95ad.png)

### request body

![request body](https://user-images.githubusercontent.com/76945789/204947636-eae02c87-1995-48b9-9575-8ef925dafd47.png)



## client

still in work, is a react-app for connecting to the server for autocompletion while typing
![client](https://user-images.githubusercontent.com/76945789/203187237-31a110a3-c340-4995-a24a-8be634f8c587.png)




## references
[gpt and casual language models](https://huggingface.co/transformers/v2.0.0/examples.html#gpt-2-gpt-and-causal-language-modeling)
