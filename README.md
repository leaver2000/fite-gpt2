# Forecast integrated TAF environment (FITE)

> ✨ Bootstrapped with Create Snowpack App (CSA).

## train the model
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
## limitations:

- TAF data does not include year and month in the body of the text.  So there is no way for the model to learn seasonal patterns like lake effect snow or thunderstorms in the summer.  This could be solved by adding the year and month to the TAF body text.

- The model reads left to right and with the TX/TN groups at the end of the forecast, the model is not able to infer precipitation types or icing conditions.  This could be solved by moving the TX/TN groups to the beginning of the forecast.

## Usage
```  python
Python 3.10.6 (main, Nov  2 2022, 18:53:38) [GCC 11.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import model
>>> pipeline = model.pipeline("gpt2", "taf")
>>> pipeline.generate_forecast("TAF KBLV 140100Z 31015KT 8000")
[['TAF KBLV 140100Z 31015KT 8000 -SHRA BKN010 OVC020 QNH2994INS', 'BECMG 1402/1403 VRB06KT 9999 NSW BKN020 QNH3004INS TX18/1319Z TN11/1410Z']]
```

## TODO:

- [ ] add year and month to TAF body text
- [ ] move TX/TN groups to beginning of forecast
- [ ] add more locations to dataset
- [ ] add more TAFs from each location to dataset




## CLIENT: Available Scripts

### npm start

Runs the app in the development mode.
Open <http://localhost:8080> to view it in the browser.

The page will reload if you make edits.
You will also see any lint errors in the console.

### npm run build

Builds a static copy of your site to the `build/` folder.
Your app is ready to be deployed!

**For the best production performance:** Add a build bundler plugin like "@snowpack/plugin-webpack" to your `snowpack.config.mjs` config file.

### npm test

Launches the application test runner.
Run with the `--watch` flag (`npm test -- --watch`) to run in interactive watch mode.

## SERVER: Available Scripts

### npm run server

starts the FastAPI Uvicorn server

#### routes

- `/` - returns a simple message
- `generate/` - returns a TAF for a given station

## NLP Model

### Environment

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

### usage
``` bash 
# give the model the start of a TAF and it will encode the first line
(fite-venv) ➜  fite-gpt2 git:(development) ✗ python -m model.gpt2 main --text "TAF KBLV 151100Z 1506/1511 27010G20KT"
[{'generated_text': 'TAF KBLV 151100Z 1506/1511 27010G20KT 9999 BKN020 QNH3029INS'}]
# increasing the winds the model knows to encode the turbulence block, but has not been trained sufficiently to to understand the usage of NSW
# it also decides to lower the ceiling and encode a icing; 
# NOTE: that the icing appears to be encoded properly, but because at this point the model has no information about temperature, 
# so it struggles to encode the icing properly 
(fite-venv) ➜  fite-gpt2 git:(development) ✗ python -m model.gpt2 main --text "TAF KBLV 151100Z 1506/1511 27020G35KT"
[{'generated_text': 'TAF KBLV 151100Z 1506/1511 27020G35KT 9999 NSW OVC007 620079 510004 QNH2994INS'}]
# using the text generated above we can prompt the model to create a TEMPO line. Which it decides to encode a reduced ceiling and visibility and -RA
(fite-venv) ➜  fite-gpt2 git:(development) ✗ python -m model.gpt2 main --text "TAF KBLV 151100Z 1506/1511 27020G35KT 9999 NSW OVC007 620079 510004 QNH2994INS\n TEMPO 1506/"
[{'generated_text': 'TAF KBLV 151100Z 1506/1511 27020G35KT 9999 NSW OVC007 620079 510004 QNH2994INS\\n TEMPO 1506/1510 8000 -RA BKN008'}]
# given a prompt with a visibility reduction and the model will encode present weather and lower ceilings
(fite-venv) ➜  fite-gpt2 git:(development) ✗ python -m model.gpt2 main --text "TAF KBLV 151100Z 1506/1511 27020G35KT 8000"
[{'generated_text': 'TAF KBLV 151100Z 1506/1511 27020G35KT 8000 -RA BKN008 OVC015 QNH2994INS'}]
```

![image](https://user-images.githubusercontent.com/76945789/203183599-ba4adad0-d87b-407a-94ac-d9acb2c19d08.png)


## references
[gpt and casual language models](https://huggingface.co/transformers/v2.0.0/examples.html#gpt-2-gpt-and-causal-language-modeling)