import React from "react";
import { FITEContext } from "./context";
import type { FITEState } from "./context";

function createUrls(baseUrl: string | URL) {
  return {
    models: new URL("/models", baseUrl),
    strategies: new URL("/strategies", baseUrl),
    generate: new URL("/generate", baseUrl),
  };
}

function updateSearchParams(generateURL: URL, { model, strategy }: FITEState) {
  generateURL.searchParams.set("model", model);
  generateURL.searchParams.set("strategy", strategy);
}
// the metadata interface is used to pass additional information to the api
// prefixing the TAF with temperature information gives context to the model
interface MetaData {
  TX: number;
  TN: number;
  issueDatetime: Date;
}

function packagePayload(text: string, metadata: MetaData) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, metadata }),
  };
}

function useFITE() {
  const { __setState, ...state } = React.useContext(FITEContext);
  //  create a url object from the baseUrl for the various api calls
  const urls = React.useMemo(() => createUrls(state.apiUrl), [state.apiUrl]);
  // if the model or strategy changes, update the generate url query params
  React.useEffect(() => updateSearchParams(urls.generate, state), [urls.generate, state.model, state.strategy]);
  // it is unlikely that the model-options or strategy-options will change so this effect should only run once
  React.useEffect(() => {
    const options = { method: "GET" };
    // fetch the model and strategy options from the server
    const results = [urls.strategies, urls.models].map((url) =>
      fetch(url, options).then((res) => res.json())
    ) as Promise<string[]>[];
    // update the state with the results of the fetches
    Promise.all(results).then(([strategyOptions, modelOptions]) =>
      __setState(({ model, strategy, ...state }) => {
        // if the model or strategy is not in the list of options, set it to the first option
        if (!modelOptions.includes(model) || !model) {
          // raise a warning if the model is not in the list of modelOptions
          console.warn(`model ${model} not in list of modelOptions options ${modelOptions}, using first model in list`);
          model = modelOptions[0];
        }
        if (!strategyOptions.includes(strategy) || !strategy) {
          // raise a warning if the strategy is not in the list of strategyOptions
          console.warn(
            `strategy ${strategy} not in list of strategyOptions options ${strategyOptions}, using first strategy in list`
          );
          strategy = strategyOptions[0];
        }
        return { ...state, model, strategy, strategyOptions, modelOptions };
      })
    );
  }, [urls.models, urls.strategies]);
  // the dispatch function for the context
  const setPartialState = React.useCallback<(partialState: Partial<FITEState>) => void>(
    (partialState) => __setState((prevState) => ({ ...prevState, ...partialState })),
    [__setState]
  );
  // post the textArea value to the api/generate and return the result as a Promise<string[]>
  const generateText = React.useCallback<(text: string) => Promise<string[]>>(
    async (text) => {
      // if some empty text is passed, return an empty array
      if (text.trim().length === 0) return [];
      const payload = packagePayload(
        text,
        // TODO: add metadata to the payload
        // max and min temperatures could be set by the forecaster
        { TX: 0, TN: 0, issueDatetime: new Date() }
      );

      const res = await fetch(urls.generate, payload);
      return await res.json();
    },
    [urls.generate]
  );

  return { setPartialState, generateText, ...state };
}

export default useFITE;
