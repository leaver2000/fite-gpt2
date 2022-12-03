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

function useFite() {
  const { __setState, ...state } = React.useContext(FITEContext);
  const urls = React.useMemo(
    //  create a url object from the baseUrl for the various api calls
    () => createUrls(state.apiUrl),
    [state.apiUrl]
  );
  React.useEffect(() => {
    // if the model or strategy changes, update the generate url
    const { model, strategy } = state;
    urls.generate.searchParams.set("model", model);
    urls.generate.searchParams.set("strategy", strategy);
  }, [state.model, state.strategy]);

  const dispatchState = React.useCallback(
    (partialState: Partial<FITEState>) =>
    __setState((prevState: FITEState) => ({...prevState,...partialState})), // prettier-ignore
    [__setState]
  );

  const generateText = React.useCallback(
    (text: string, model: string, strategy: string) => {
      const payload = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      };

      return fetch(urls.generate, payload).then((res) => res.json()) as Promise<string[]>;
    },
    [urls.generate]
  );
  React.useEffect(() => {
    const options = { method: "GET" };
    const results = [urls.strategies, urls.models].map((url) =>
      fetch(url, options).then((res) => res.json())
    ) as Promise<string[]>[];
    //
    Promise.all(results).then(([strategies, models]) => dispatchState({ strategies, models }));
  }, [urls.models, urls.strategies]);

  React.useEffect(() => {
    const { models, strategies } = state;
    // setting the default model and strategy`
    if (models && models.length > 0) dispatchState({ model: models[0] });
    if (strategies && strategies.length > 0) dispatchState({ strategy: strategies[0] });
  }, [state.models, state.strategies]);

  return { dispatchState, generateText, ...state };
}

export default useFite;
