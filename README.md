# Forecast integrated TAF environment (FITE)

> ✨ Bootstrapped with Create Snowpack App (CSA).

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
- `/api/taf/{station_id}` - returns a TAF for a given station

## NLP Model

### usage
``` bash
python -m model.gpt2 --text "TAF KRAP"
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
[{'generated_text': 'TAF KRAP 08006KT 9999 FEW020 QNH2994INS\nBECMG 08006KT 9999 FEW030 QNH2994INS\nBECMG 08006KT 9999 FEW030 QNH2994INS\nBECMG 08006KT 9999 FEW030 QNH2994INS\nBECMG 08006KT 9999 FEW030 QNH2994INS\nBECMG 08006KT 9999 FEW030 QNH2994INS\nBECMG 08006KT 9999 FEW'}]
```

## client

![image](https://user-images.githubusercontent.com/76945789/203187237-31a110a3-c340-4995-a24a-8be634f8c587.png)

## server
![image](https://user-images.githubusercontent.com/76945789/203183599-ba4adad0-d87b-407a-94ac-d9acb2c19d08.png)
