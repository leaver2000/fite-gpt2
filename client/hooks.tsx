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

function useFite() {
  const { __setState, ...state } = React.useContext(FITEContext);
  //  create a url object from the baseUrl for the various api calls
  const urls = React.useMemo(() => createUrls(state.apiUrl), [state.apiUrl]);
  // if the model or strategy changes, update the generate url
  React.useEffect(() => updateSearchParams(urls.generate, state), [urls.generate, state.model, state.strategy]);
  // unlike in the current application for this effect to occur more than once
  // the model api and or strategy api would have to change
  React.useEffect(() => {
    const options = { method: "GET" };
    const results = [urls.strategies, urls.models].map((url) =>
      fetch(url, options).then((res) => res.json())
    ) as Promise<string[]>[];
    //
    Promise.all(results).then(([strategies, models]) => dispatchState({ strategies, models }));
  }, [urls.models, urls.strategies]);
  // settings a default value for the model and strategy
  React.useEffect(() => {
    const { models, strategies } = state;
    // setting the default model and strategy if one is not set
    if (!state.model && models && models.length > 0) dispatchState({ model: models[0] });
    if (!state.strategy && strategies && strategies.length > 0) dispatchState({ strategy: strategies[0] });
  }, [state.models, state.strategies]);
  // the dispatch function for the context
  const dispatchState = React.useCallback(
    (partialState: Partial<FITEState>) =>
    __setState((prevState: FITEState) => ({...prevState,...partialState})), // prettier-ignore
    [__setState]
  );
  // post the textArea value to the generate url
  const generateText = React.useCallback(
    (text: string) => {
      const payload = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      };

      return fetch(urls.generate, payload).then((res) => res.json()) as Promise<string[]>;
    },
    [urls.generate]
  );

  return { dispatchState, generateText, ...state };
}

export default useFite;
