import React from "react";
import { FITEContext } from "./context";
import type { FITEState } from "./context";

interface APIState {
  models: string[];
  strategies: string[];
}
const defaultAPIState = { models: [], strategies: [] };

function useApi(apiUrl: string) {
  const endPoints = ["strategies", "models"];
  const options = { method: "GET" };
  const [state, setState] = React.useState<APIState>(defaultAPIState);

  React.useEffect(() => {
    const results = endPoints.map((key) => fetch(`${apiUrl}/${key}/`, options).then((res) => res.json()));
    Promise.all(results).then(([strategies, models]) => setState({ strategies, models }));
  }, [apiUrl]);

  const generateText = React.useCallback(
    (text: string, model: string, strategy: string) => {
      const url = `${apiUrl}/generate/${model}?strategy=${strategy}`;
      const options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      };
      return fetch(url, options).then((res) => res.json()) as Promise<string[]>;
    },
    [state]
  );

  return { generateText, ...state };
}

function useFite() {
  const { __setState, ...state } = React.useContext(FITEContext);
  const api = useApi(state.apiUrl);
  // const { text, model, strategy } = state;
  const { models, strategies } = api;

  const dispatchState = React.useCallback(
    (partialState: Partial<FITEState>) =>
      __setState((prevState: FITEState) => ({
        ...prevState,
        ...partialState,
      })),
    [__setState]
  );

  React.useEffect(() => {
    // setting the default model and strategy`
    if (models.length > 0) dispatchState({ model: models[0] });
    if (strategies.length > 0) dispatchState({ strategy: strategies[0] });
  }, [models, strategies]);

  return { dispatchState, ...api, ...state };
}

export default useFite;
